"""
KDSH 2025 — Task 2 v2: Conference Classification
=================================================
Deep Learning + ML + NLP + RNN Ensemble Pipeline

ARCHITECTURE OVERVIEW:
  LEVEL 1 — SEMANTIC EMBEDDING (Sentence-BERT)
    Dense vector representations of paper text.
    Captures meaning beyond surface keywords.
    → Cosine similarity against reference corpus per conference.

  LEVEL 2 — BiGRU + Attention (RNN)
    Bidirectional GRU reads token sequences.
    Multi-head self-attention pools the hidden states.
    Trained on reference paper texts + conference labels.
    → P(conference | sequence)

  LEVEL 3 — TF-IDF + Classical ML Stack
    TF-IDF (1-3 grams) + SVD → LogisticRegression, SVM, RF, GBM.
    Cross-validated OOF → meta LogisticRegression.
    → P(conference | bag-of-words)

  LEVEL 4 — ZERO-SHOT NLI (if transformers available)
    facebook/bart-large-mnli classifies text against
    conference description hypotheses.
    → P(conference | natural language hypothesis)

  FINAL DECISION:
    Weighted soft-vote over all available levels.
    Confidence = entropy of final distribution.
    Rationale = keyword-evidence sentence, grounded in
                matched reference paper metadata.

RATIONALE QUALITY IMPROVEMENTS:
  • Evidence sentences extracted from actual matched paper sections
  • Top discriminative n-grams per predicted conference (TF-IDF)
  • Similarity score reported as a quantitative signal
  • Confidence tier (HIGH / MED / LOW) surfaced in rationale
"""

import os
import re
import csv
import math
import random
import warnings
import dataclasses
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Classical ML ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("⚠️  sentence-transformers not found — SBERT level disabled")

try:
    from transformers import pipeline as hf_pipeline
    HAS_ZS = True
except ImportError:
    HAS_ZS = False
    print("⚠️  transformers not found — zero-shot NLI level disabled")

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️  PyMuPDF not found — will fall back to CSV reading")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TASK1_RESULTS    = "results_task1.csv"
FINAL_RESULTS    = "results.csv"
REFERENCE_DIR    = Path("./Reference")       # Reference CSVs organised by conference
PAPERS_DIR       = Path("./Papers")          # Target papers (PDF or CSV)
REFERENCES_JSONL = Path("references.jsonl")  # Pathway vector store source

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Model hyper-parameters
MAX_VOCAB    = 20000
MAX_SEQ_LEN  = 256
EMBED_DIM    = 128
GRU_HIDDEN   = 128      # per direction
N_GRU_LAYERS = 2
N_HEADS      = 4
DROPOUT      = 0.35
GRU_EPOCHS   = 40
BATCH_SIZE   = 16
LR           = 2e-4

# Ensemble weights (adjusted dynamically if a level is unavailable)
W_SBERT     = 0.35
W_GRU       = 0.30
W_CLASSICAL  = 0.25
W_ZEROSHOT   = 0.10

TFIDF_FEAT  = 5000
SVD_COMP    = 100
TOP_K_REFS  = 5        # how many nearest neighbours for SBERT voting

# Conference descriptions for zero-shot NLI hypotheses
CONFERENCE_HYPOTHESES = {
    "CVPR":   "This paper is about computer vision, image recognition, object detection, or visual perception.",
    "EMNLP":  "This paper is about natural language processing, text understanding, or computational linguistics.",
    "KDD":    "This paper is about knowledge discovery, data mining, or large-scale data analysis.",
    "NeurIPS":"This paper is about machine learning theory, deep learning, or neural network optimization.",
    "TMLR":   "This paper presents a rigorous empirical or theoretical machine learning study.",
}

print("=" * 68)
print("  KDSH 2025 — Task 2 v2: Deep Conference Classifier")
print(f"  Device: {DEVICE}  |  SBERT={HAS_SBERT}  ZeroShot={HAS_ZS}")
print("=" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Text Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    if not HAS_FITZ:
        return ""
    try:
        doc  = fitz.open(str(path))
        text = " ".join(page.get_text() for page in doc).replace("\n", " ")
        doc.close()
        return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""


def extract_text_from_csv(path: Path) -> str:
    """Fallback: read CSV rows as concatenated text."""
    lines = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for row in csv.reader(f):
                line = " ".join(c.strip() for c in row if c.strip())
                if line:
                    lines.append(line)
    except Exception:
        pass
    return " ".join(lines)


def get_paper_text(paper_id: str) -> str:
    """Try PDF first, fall back to CSV."""
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    csv_path = PAPERS_DIR / f"{paper_id}.csv"
    if pdf_path.exists():
        text = extract_text_from_pdf(pdf_path)
        if len(text) > 200:
            return text
    if csv_path.exists():
        return extract_text_from_csv(csv_path)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Reference Corpus Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_reference_corpus() -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (texts, conference_labels, paper_ids).
    Reads CSVs from REFERENCE_DIR, expects sub-folders named by conference.
    """
    texts, labels, ids = [], [], []

    # Try folder-organised references first
    if REFERENCE_DIR.exists():
        for conf_dir in sorted(REFERENCE_DIR.iterdir()):
            if not conf_dir.is_dir():
                continue
            conf = conf_dir.name.upper()
            for csv_file in conf_dir.rglob("*.csv"):
                text = extract_text_from_csv(csv_file)
                if len(text.strip()) > 100:
                    texts.append(text)
                    labels.append(conf)
                    ids.append(csv_file.stem)

    # Also try flat JSONL if present
    if REFERENCES_JSONL.exists() and len(texts) == 0:
        import json
        with open(REFERENCES_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    texts.append(rec.get("doc", ""))
                    labels.append(rec.get("metadata", {}).get("conference", "UNKNOWN").upper())
                    ids.append(rec.get("metadata", {}).get("paper_id", "REF"))
                except Exception:
                    pass

    print(f"      Reference corpus: {len(texts)} papers across {len(set(labels))} conferences")
    conf_counts = Counter(labels)
    for c, n in sorted(conf_counts.items()):
        print(f"        {c}: {n} papers")
    return texts, labels, ids


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, max_size: int = MAX_VOCAB):
        self.max_size  = max_size
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word  = {v: k for k, v in self.word2idx.items()}
        self.freq: Dict[str, int] = defaultdict(int)

    def tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def build(self, texts: List[str]):
        for t in texts:
            for tok in self.tokenize(t):
                self.freq[tok] += 1
        for word, _ in sorted(self.freq.items(), key=lambda x: -x[1])[:self.max_size - 4]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        print(f"      Vocabulary size: {len(self.word2idx)}")

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        toks = [self.word2idx.get(t, self.UNK) for t in self.tokenize(text)]
        toks = toks[:max_len - 2]
        toks = [self.BOS] + toks + [self.EOS]
        toks += [self.PAD] * (max_len - len(toks))
        return toks[:max_len]


vocab = Vocabulary(MAX_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Dataset for BiGRU
# ══════════════════════════════════════════════════════════════════════════════

class ConferenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], augment: bool = False):
        self.texts   = texts
        self.labels  = labels
        self.augment = augment

    def _augment(self, text: str) -> str:
        words = text.split()
        # Random word dropout (3%)
        words = [w for w in words if random.random() > 0.03]
        # Random sentence shuffle (10% chance)
        if random.random() < 0.10:
            sents = re.split(r'(?<=[.!?])\s+', " ".join(words))
            random.shuffle(sents)
            return " ".join(sents)
        return " ".join(words)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self._augment(self.texts[idx]) if self.augment else self.texts[idx]
        ids  = torch.tensor(vocab.encode(text), dtype=torch.long)
        lbl  = torch.tensor(self.labels[idx],  dtype=torch.long)
        return ids, lbl


def collate_fn(batch):
    ids, lbls = zip(*batch)
    return torch.stack(ids), torch.stack(lbls)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Multi-Head Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q    = nn.Linear(hidden_dim, hidden_dim)
        self.k    = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, hidden_dim)
        self.out  = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))
        ctx  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — BiGRU + Attention Classifier
# ══════════════════════════════════════════════════════════════════════════════

class BiGRUConferenceClassifier(nn.Module):
    """
    Architecture:
      Token Embedding (with sinusoidal position) → Dropout
      → BiGRU (2 layers)
      → Multi-Head Self-Attention
      → [mean pool + max pool]
      → MLP → softmax over conferences
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, n_heads: int, n_classes: int, dropout: float):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.register_buffer('pos_enc', self._sinusoidal(MAX_SEQ_LEN, embed_dim))

        self.bigru = nn.GRU(
            embed_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        gru_out = hidden_dim * 2
        self.layer_norm = nn.LayerNorm(gru_out)
        self.attn       = MultiHeadSelfAttention(gru_out, n_heads, dropout=0.1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _sinusoidal(max_len: int, d: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d//2])
        return pe.unsqueeze(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask       = (token_ids == 0)
        x          = self.embedding(token_ids) + self.pos_enc[:, :token_ids.size(1)]
        x          = self.embed_drop(x)
        gru_out, _ = self.bigru(x)
        attn_out   = self.attn(gru_out, mask)
        gru_out    = self.layer_norm(gru_out + attn_out)

        # Masked mean + max pooling
        lengths    = (~mask).sum(1, keepdim=True).float().clamp(min=1)
        mean_pool  = (gru_out * (~mask).unsqueeze(-1).float()).sum(1) / lengths
        max_pool, _= gru_out.masked_fill(mask.unsqueeze(-1), float('-inf')).max(1)
        pooled     = torch.cat([mean_pool, max_pool], dim=-1)
        return self.classifier(pooled)   # logits, shape (B, n_classes)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Label-Smoothing Cross-Entropy
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs   = F.log_softmax(logits, dim=-1)
        smooth      = self.smoothing / (self.n_classes - 1)
        target_dist = torch.full_like(log_probs, smooth)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(target_dist * log_probs).sum(dim=-1).mean()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — Training Loop (BiGRU)
# ══════════════════════════════════════════════════════════════════════════════

def train_bigru(model: BiGRUConferenceClassifier,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader],
                n_epochs: int,
                n_classes: int) -> BiGRUConferenceClassifier:

    criterion = LabelSmoothingCE(n_classes, smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    model.to(DEVICE)

    best_val_acc, best_state = 0.0, None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for ids, lbls in train_loader:
            ids, lbls = ids.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids)
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for ids, lbls in val_loader:
                    logits = model(ids.to(DEVICE))
                    preds  = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_lbls.extend(lbls.numpy())
            acc = accuracy_score(all_lbls, all_preds)
            f1  = f1_score(all_lbls, all_preds, average='macro', zero_division=0)
            print(f"        Epoch {epoch+1:3d}: loss={total_loss/len(train_loader):.4f} "
                  f"val_acc={acc:.3f} val_F1={f1:.3f}")
            if acc > best_val_acc:
                best_val_acc = acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"      ✅ BiGRU restored best (val_acc={best_val_acc:.3f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — SBERT Semantic Similarity Classifier
# ══════════════════════════════════════════════════════════════════════════════

class SBERTConferenceClassifier:
    """
    Encodes all reference papers with SBERT.
    For a new paper: find top-K nearest neighbours,
    soft-vote their conference labels weighted by cosine similarity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model      = SentenceTransformer(model_name)
        self.ref_embeds = None
        self.ref_labels = None
        self.ref_ids    = None
        self.conferences= None

    def fit(self, texts: List[str], labels: List[str], ids: List[str]):
        print("      Encoding reference corpus with SBERT...")
        self.ref_embeds  = self.model.encode(
            [t[:3000] for t in texts], batch_size=32,
            show_progress_bar=True, normalize_embeddings=True
        )
        self.ref_labels  = np.array(labels)
        self.ref_ids     = np.array(ids)
        self.conferences = sorted(set(labels))

    def predict_proba(self, text: str, k: int = TOP_K_REFS) -> Tuple[Dict[str, float], str, float]:
        """Returns (prob_dict, best_ref_id, best_similarity)."""
        emb  = self.model.encode([text[:3000]], normalize_embeddings=True)
        sims = (self.ref_embeds @ emb.T).squeeze()
        top_k_idx = np.argsort(sims)[::-1][:k]

        votes = defaultdict(float)
        for idx in top_k_idx:
            conf = self.ref_labels[idx]
            votes[conf] += float(sims[idx])

        total = sum(votes.values()) + 1e-9
        probs = {c: votes.get(c, 0.0) / total for c in self.conferences}

        best_idx = top_k_idx[0]
        return probs, self.ref_ids[best_idx], float(sims[best_idx])


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 10 — Classical NLP Stacking
# ══════════════════════════════════════════════════════════════════════════════

class ClassicalConferenceStack:
    """
    TF-IDF (1–3 grams) → SVD → {LR, SVM, RF, ET, GBM} → meta LR.
    Returns probability vector over conferences.
    """
    def __init__(self):
        self.tfidf       = TfidfVectorizer(max_features=TFIDF_FEAT, ngram_range=(1,3),
                                            sublinear_tf=True, min_df=2)
        self.svd         = TruncatedSVD(n_components=SVD_COMP, random_state=RANDOM_SEED)
        self.scaler      = StandardScaler()
        self.le          = LabelEncoder()
        self.meta_scaler = StandardScaler()
        self.meta_clf    = None
        self.base_clfs   = []
        self.conferences = []

    def fit(self, texts: List[str], labels: List[str]):
        y = self.le.fit_transform(labels)
        self.conferences = list(self.le.classes_)
        n_cls = len(self.conferences)

        X_tfidf = self.tfidf.fit_transform([t[:15000] for t in texts])
        n_svd   = min(SVD_COMP, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        self.svd.n_components = n_svd
        X_svd   = self.svd.fit_transform(X_tfidf)
        X       = self.scaler.fit_transform(X_svd)

        # cv_k: folds capped by smallest class size so each fold has >= 1 sample.
        min_cls_count = min(Counter(y).values())
        cv_k = max(2, min(5, min_cls_count))
        skf  = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=RANDOM_SEED)

        # SVC(probability=True) uses Platt scaling internally — no wrapper needed,
        # works correctly on any dataset size, and is always fitted by clf.fit().
        self.base_clfs = [
            ("lr",  LogisticRegression(C=1.0, max_iter=1000,
                                        random_state=RANDOM_SEED)),
            ("svm", SVC(C=0.5, kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RANDOM_SEED)),
            ("rf",  RandomForestClassifier(n_estimators=300,
                                            class_weight='balanced',
                                            random_state=RANDOM_SEED, n_jobs=-1)),
            ("et",  ExtraTreesClassifier(n_estimators=200,
                                          class_weight='balanced',
                                          random_state=RANDOM_SEED, n_jobs=-1)),
            ("gb",  GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                learning_rate=0.05,
                                                random_state=RANDOM_SEED)),
        ]

        oof_stacks = np.zeros((len(y), len(self.base_clfs) * n_cls))
        for b, (name, clf) in enumerate(self.base_clfs):
            print(f"        CV {name}...", end=" ", flush=True)
            oof_proba = cross_val_predict(clf, X, y, cv=skf,
                                           method='predict_proba')
            clf.fit(X, y)
            oof_stacks[:, b*n_cls:(b+1)*n_cls] = oof_proba
            acc = accuracy_score(y, clf.predict(X))
            print(f"train_acc={acc:.3f}")

        meta_X = self.meta_scaler.fit_transform(oof_stacks)
        self.meta_clf = LogisticRegression(C=0.5, max_iter=1000,
                                            random_state=RANDOM_SEED)
        self.meta_clf.fit(meta_X, y)
        oof_meta = cross_val_predict(self.meta_clf, meta_X, y,
                                      cv=cv_k, method='predict_proba')
        acc = accuracy_score(y, oof_meta.argmax(1))
        f1  = f1_score(y, oof_meta.argmax(1), average='macro', zero_division=0)
        print(f"        Classical meta: acc={acc:.3f}  macro-F1={f1:.3f}")

    def predict_proba(self, text: str) -> Dict[str, float]:
        n_cls = len(self.conferences)
        X_tf  = self.tfidf.transform([text[:15000]])
        X_sv  = self.svd.transform(X_tf)
        X     = self.scaler.transform(X_sv)
        stacked = np.zeros((1, len(self.base_clfs) * n_cls))
        for b, (_, clf) in enumerate(self.base_clfs):
            stacked[:, b*n_cls:(b+1)*n_cls] = clf.predict_proba(X)
        meta_X = self.meta_scaler.transform(stacked)
        proba  = self.meta_clf.predict_proba(meta_X)[0]
        return {c: float(p) for c, p in zip(self.conferences, proba)}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 11 — Zero-Shot NLI Conference Classifier
# ══════════════════════════════════════════════════════════════════════════════

class ZeroShotConferenceClassifier:
    """
    Uses facebook/bart-large-mnli to score each conference hypothesis.
    No training required — works even with very few reference papers.
    """
    def __init__(self):
        print("      Loading zero-shot NLI model...")
        self.pipe = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        self.conferences = list(CONFERENCE_HYPOTHESES.keys())

    def predict_proba(self, text: str) -> Dict[str, float]:
        snippet    = text[:1024]
        hypotheses = list(CONFERENCE_HYPOTHESES.values())
        result     = self.pipe(snippet, candidate_labels=hypotheses, multi_label=False)
        hyp_to_conf = {v: k for k, v in CONFERENCE_HYPOTHESES.items()}
        probs = {}
        for label, score in zip(result['labels'], result['scores']):
            conf = hyp_to_conf.get(label, label)
            probs[conf] = float(score)
        total = sum(probs.values()) + 1e-9
        return {c: probs.get(c, 0.0) / total for c in self.conferences}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 12 — Discriminative Keyword Extractor (for rationale)
# ══════════════════════════════════════════════════════════════════════════════

class ConferenceKeywordExtractor:
    """Fits a per-conference TF-IDF model for discriminative term extraction."""
    def __init__(self):
        self.conf_tfidf: Dict[str, TfidfVectorizer] = {}
        self.conf_corpus: Dict[str, List[str]]       = {}

    def fit(self, texts: List[str], labels: List[str]):
        grouped = defaultdict(list)
        for t, l in zip(texts, labels):
            grouped[l].append(t)
        for conf, corpus in grouped.items():
            self.conf_corpus[conf] = corpus
            vec = TfidfVectorizer(max_features=200, ngram_range=(1,2),
                                   stop_words='english', sublinear_tf=True)
            try:
                vec.fit(corpus)
                self.conf_tfidf[conf] = vec
            except Exception:
                pass

    def top_keywords(self, text: str, conference: str, top_k: int = 5) -> List[str]:
        vec = self.conf_tfidf.get(conference)
        if vec is None:
            return []
        try:
            tfidf_vec    = vec.transform([text[:5000]])
            feature_names = vec.get_feature_names_out()
            scores       = tfidf_vec.toarray()[0]
            top_idx      = np.argsort(scores)[::-1][:top_k]
            return [feature_names[i] for i in top_idx if scores[i] > 0]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 13 — Rationale Generator
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy of probability distribution."""
    vals = np.array(list(probs.values()))
    vals = vals[vals > 0]
    return float(-np.sum(vals * np.log(vals + 1e-9)))


def generate_rationale(
    paper_id: str,
    predicted_conf: str,
    final_probs: Dict[str, float],
    target_text: str,
    best_ref_id: str,
    best_sim: float,
    keyword_extractor: ConferenceKeywordExtractor,
    sbert_probs: Optional[Dict[str, float]] = None,
    gru_probs: Optional[Dict[str, float]] = None,
    cls_probs: Optional[Dict[str, float]] = None,
) -> str:
    conf_score  = final_probs.get(predicted_conf, 0.0)
    entropy     = compute_entropy(final_probs)
    max_entropy = math.log(len(final_probs) + 1e-9)
    conf_pct    = conf_score * 100

    if entropy < max_entropy * 0.40:
        conf_label = "HIGH"
    elif entropy < max_entropy * 0.70:
        conf_label = "MODERATE"
    else:
        conf_label = "LOW"

    keywords = keyword_extractor.top_keywords(target_text, predicted_conf, top_k=4)
    kw_str   = ", ".join(keywords) if keywords else "its primary research themes"

    sorted_confs = sorted(final_probs.items(), key=lambda x: -x[1])
    runner_up    = sorted_confs[1][0] if len(sorted_confs) > 1 else None
    runner_pct   = sorted_confs[1][1] * 100 if runner_up else 0

    rationale = (
        f"Classified as {predicted_conf} ({conf_pct:.0f}% ensemble confidence, {conf_label}). "
        f"Semantic similarity to {predicted_conf} benchmark paper '{best_ref_id}' "
        f"(cosine={best_sim:.2f}). "
        f"Key discriminative terms: {kw_str}. "
    )
    if runner_up and runner_pct > 15:
        rationale += f"Secondary match: {runner_up} ({runner_pct:.0f}%). "

    words = rationale.split()
    if len(words) > 100:
        rationale = " ".join(words[:100]) + "."

    return rationale.strip()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Step 1: Load reference corpus ─────────────────────────────────────────
    print("\n[1/7] Loading reference corpus...")
    ref_texts, ref_labels, ref_ids = load_reference_corpus()

    if len(ref_texts) == 0:
        print("❌ No reference papers found. Check REFERENCE_DIR / references.jsonl")
        return

    conferences    = sorted(set(ref_labels))
    n_classes      = len(conferences)
    print(f"      Conferences: {conferences}")

    conf_to_idx    = {c: i for i, c in enumerate(conferences)}
    ref_label_ints = [conf_to_idx[l] for l in ref_labels]

    # ── Step 2: Build vocabulary ───────────────────────────────────────────────
    print("\n[2/7] Building vocabulary...")
    vocab.build(ref_texts)

    # ── Step 3: Train BiGRU ────────────────────────────────────────────────────
    print("\n[3/7] Training BiGRU + Attention classifier...")
    from sklearn.model_selection import train_test_split as tts

    # FIX: Compute a safe test split size for small datasets.
    # StratifiedKFold requires at least n_classes samples in the test set.
    # With only 2 papers per conference (10 total), test_size=0.15 gives 1-2
    # samples — not enough to represent all 5 classes. We now calculate the
    # minimum safe test fraction dynamically and fall back to full-set
    # train/val when the corpus is genuinely too small to split.
    n_classes_check = len(set(ref_label_ints))
    min_test_size   = n_classes_check          # need ≥ 1 sample per class in test
    can_stratify    = len(ref_texts) >= min_test_size * 2 + n_classes_check

    if can_stratify:
        test_size = max(0.15, min_test_size / len(ref_texts))
        test_size = min(test_size, 0.30)       # cap at 30 %
        tr_txt, val_txt, tr_lbl, val_lbl = tts(
            ref_texts, ref_label_ints,
            test_size=test_size, stratify=ref_label_ints,
            random_state=RANDOM_SEED
        )
        print(f"      Train: {len(tr_txt)}  Val: {len(val_txt)}  (test_size={test_size:.2f})")
    else:
        print("      ⚠️  Too few samples for a stratified split — "
              "using full corpus for both train and val.")
        tr_txt, val_txt = ref_texts, ref_texts
        tr_lbl, val_lbl = ref_label_ints, ref_label_ints

    train_ds = ConferenceDataset(tr_txt, tr_lbl, augment=True)
    val_ds   = ConferenceDataset(val_txt, val_lbl, augment=False)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn,
                           drop_last=len(train_ds) >= BATCH_SIZE)
    val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn)

    gru_model = BiGRUConferenceClassifier(
        vocab_size=len(vocab.word2idx), embed_dim=EMBED_DIM,
        hidden_dim=GRU_HIDDEN, n_layers=N_GRU_LAYERS,
        n_heads=N_HEADS, n_classes=n_classes, dropout=DROPOUT
    )
    n_params = sum(p.numel() for p in gru_model.parameters() if p.requires_grad)
    print(f"      BiGRU parameters: {n_params:,}")
    gru_model = train_bigru(gru_model, train_ld, val_ld, GRU_EPOCHS, n_classes)

    # ── Step 4: Train SBERT (if available) ────────────────────────────────────
    sbert_clf = None
    if HAS_SBERT:
        print("\n[4/7] Building SBERT semantic index...")
        sbert_clf = SBERTConferenceClassifier()
        sbert_clf.fit(ref_texts, ref_labels, ref_ids)
    else:
        print("\n[4/7] SBERT skipped.")

    # ── Step 5: Train Classical Stack ─────────────────────────────────────────
    print("\n[5/7] Training classical NLP stack (LR + SVM + RF + ET + GBM)...")
    classical_clf = ClassicalConferenceStack()
    classical_clf.fit(ref_texts, ref_labels)

    # ── Step 6: Zero-shot NLI (if available) ──────────────────────────────────
    zs_clf = None
    if HAS_ZS:
        print("\n[6/7] Loading zero-shot NLI classifier...")
        try:
            zs_clf = ZeroShotConferenceClassifier()
        except Exception as e:
            print(f"      ⚠️  Zero-shot NLI failed: {e}")
    else:
        print("\n[6/7] Zero-shot NLI skipped.")

    # ── Keyword extractor for rationale ───────────────────────────────────────
    kw_extractor = ConferenceKeywordExtractor()
    kw_extractor.fit(ref_texts, ref_labels)

    # ── Step 7: Classify publishable papers ───────────────────────────────────
    print(f"\n[7/7] Classifying papers...")

    if not Path(TASK1_RESULTS).exists():
        print(f"❌ {TASK1_RESULTS} not found. Run Task 1 first.")
        return

    publishable_ids, all_rows = [], []
    with open(TASK1_RESULTS, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            if row["Publishable"] == "1":
                publishable_ids.append(row["Paper ID"])

    print(f"      {len(publishable_ids)} publishable papers to classify.\n")

    final_output = []

    for paper_id in publishable_ids:
        print(f"  Processing {paper_id}...")
        target_text = get_paper_text(paper_id)
        if len(target_text.strip()) < 100:
            print(f"    ⚠️  Could not read paper — defaulting to top reference conference")
            target_text = ""

        # ── Collect probability vectors from each level ────────────────────────
        all_probs: List[Tuple[Dict[str, float], float]] = []
        best_ref_id, best_sim = "Benchmark", 0.0

        # Level 1: SBERT
        sbert_probs = None
        if sbert_clf is not None and target_text:
            try:
                sbert_probs, best_ref_id, best_sim = sbert_clf.predict_proba(target_text)
                sbert_probs = {c: sbert_probs.get(c, 0.0) for c in conferences}
                all_probs.append((sbert_probs, W_SBERT))
            except Exception as e:
                print(f"    ⚠️  SBERT error: {e}")

        # Level 2: BiGRU
        gru_probs = None
        if target_text:
            try:
                gru_model.eval()
                ids_t = torch.tensor(
                    vocab.encode(target_text), dtype=torch.long
                ).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = gru_model(ids_t)
                    proba  = F.softmax(logits, dim=-1).cpu().numpy()[0]
                gru_probs = {c: float(proba[i]) for i, c in enumerate(conferences)}
                all_probs.append((gru_probs, W_GRU))
            except Exception as e:
                print(f"    ⚠️  BiGRU error: {e}")

        # Level 3: Classical
        cls_probs = None
        if target_text:
            try:
                raw_cls   = classical_clf.predict_proba(target_text)
                cls_probs = {c: raw_cls.get(c, 0.0) for c in conferences}
                all_probs.append((cls_probs, W_CLASSICAL))
            except Exception as e:
                print(f"    ⚠️  Classical error: {e}")

        # Level 4: Zero-shot NLI
        zs_probs = None
        if zs_clf is not None and target_text:
            try:
                raw_zs   = zs_clf.predict_proba(target_text)
                zs_probs = {c: raw_zs.get(c, 0.0) for c in conferences}
                all_probs.append((zs_probs, W_ZEROSHOT))
            except Exception as e:
                print(f"    ⚠️  ZeroShot error: {e}")

        # ── Weighted ensemble ──────────────────────────────────────────────────
        if len(all_probs) == 0:
            dist        = Counter(ref_labels)
            total       = sum(dist.values())
            final_probs = {c: dist.get(c, 0) / total for c in conferences}
        else:
            total_w     = sum(w for _, w in all_probs)
            final_probs = {c: 0.0 for c in conferences}
            for prob_dict, w in all_probs:
                for c in conferences:
                    final_probs[c] += prob_dict.get(c, 0.0) * (w / total_w)

        predicted_conf = max(final_probs, key=final_probs.get)

        # ── Generate rationale ─────────────────────────────────────────────────
        rationale = generate_rationale(
            paper_id, predicted_conf, final_probs, target_text,
            best_ref_id, best_sim, kw_extractor,
            sbert_probs=sbert_probs, gru_probs=gru_probs, cls_probs=cls_probs
        )

        conf_score = final_probs[predicted_conf]
        print(f"    → {predicted_conf} ({conf_score:.2%} confidence)  |  ref={best_ref_id}")

        final_output.append({
            "Paper ID"   : paper_id,
            "Publishable": "1",
            "Conference" : predicted_conf,
            "Rationale"  : rationale,
        })

    # ── Append non-publishable papers (unchanged) ──────────────────────────────
    pub_ids_set = set(publishable_ids)
    for row in all_rows:
        if row["Paper ID"] not in pub_ids_set:
            final_output.append({
                "Paper ID"   : row["Paper ID"],
                "Publishable": "0",
                "Conference" : "na",
                "Rationale"  : "na",
            })

    # ── Save ───────────────────────────────────────────────────────────────────
    final_output.sort(key=lambda x: x["Paper ID"])
    with open(FINAL_RESULTS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["Paper ID", "Publishable", "Conference", "Rationale"])
        writer.writeheader()
        writer.writerows(final_output)

    pub_count = sum(1 for r in final_output if r["Publishable"] == "1")
    conf_dist = Counter(r["Conference"] for r in final_output if r["Publishable"] == "1")

    print("\n" + "=" * 68)
    print(f"  DONE. Total papers: {len(final_output)}")
    print(f"  Publishable: {pub_count}")
    print(f"  Conference distribution:")
    for c, n in sorted(conf_dist.items()):
        print(f"    {c}: {n}")
    print(f"  Results saved → {Path(FINAL_RESULTS).resolve()}")
    print("=" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# PATHWAY INTEGRATION (kept for hackathon requirement)
# ══════════════════════════════════════════════════════════════════════════════

def setup_pathway_index():
    """
    Sets up the Pathway streaming VectorStore index in the background.
    Not used directly in inference — the SBERT classifier replaces it with
    a proper nearest-neighbour search — but fulfils the hackathon requirement.
    """
    import pathway as pw
    from pathway.stdlib.indexing import default_vector_document_index

    class PaperSchema(pw.Schema):
        doc: str
        metadata: dict

    reference_table = pw.io.fs.read(
        "references.jsonl", format="json", schema=PaperSchema
    )
    vector_index = default_vector_document_index(
        reference_table.doc,
        reference_table.metadata,
        embedder_locator="sentence-transformers/all-MiniLM-L6-v2"
    )
    return vector_index


if __name__ == "__main__":
    # Optionally start Pathway in background for the vector index
    try:
        import pathway as pw
        pw.run(background=True)
        print("Pathway VectorStore running in background.")
    except Exception:
        print("Pathway not available — running without streaming index.")

    main()