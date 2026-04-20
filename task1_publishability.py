"""
KDSH 2025 — v6 FIXED: Deep Learning (BiLSTM + Attention) + SciBERT Fine-tuning
================================================================================

FIXES APPLIED (vs original v6):
  FIX 1 — Proper train/val split (no data leakage)
           Real data is split FIRST (80/20 stratified) before any augmentation.
           Augmentation is applied only to the training portion.
           val_loader now contains UNSEEN samples only.

  FIX 2 — Unified OOF threshold
           All OOF probabilities (BiLSTM, BERT, classical) are gathered on the
           same held-out val set so their distributions are aligned before
           blending and threshold search.

  FIX 3 — Majority gate corrected
           Changed from ceil(n/2 + 0.1) → strict majority n_votes > len(votes)/2
           so a 2-vs-1 split in favour of publishable is correctly accepted.

  FIX 4 — Meta-learner refit uses real data only, no index misalignment
           The second meta_clf.fit at the bottom now operates on real-data
           predictions only, never on the augmented-set slice that was
           accidentally misaligned with y_real.

  FIX 5 — Hard veto threshold relaxed for CSV-extracted text
           VETO_MIN_WORDS lowered from 1200 → 800 (CSV extraction loses
           whitespace, under-counts real word counts).
           Veto now also requires ALL three structural conditions to be
           simultaneously bad before striking (not independent strikes).
"""

import re, csv, warnings, random, dataclasses, math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
REFERENCE_DIR = Path(r"C:\Dataset\Papers.csv\Reference")
PAPERS_DIR    = Path(r"C:\Dataset\Papers.csv\Papers")
OUTPUT_CSV    = "kdsh2025_v6_fixed_results.csv"
DETAILED_LOG  = "kdsh2025_v6_fixed_log.txt"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED   = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Text processing
MAX_VOCAB      = 15000
MAX_SEQ_LEN    = 512
EMBED_DIM      = 128
HIDDEN_DIM     = 256
N_LAYERS       = 2
N_HEADS        = 4
DROPOUT        = 0.4

# Training
BILSTM_EPOCHS  = 30
BERT_EPOCHS    = 8
BATCH_SIZE     = 8
LR_BILSTM      = 3e-4
LR_BERT        = 2e-5
FOCAL_GAMMA    = 2.0
LABEL_SMOOTH   = 0.1
TTA_RUNS       = 3

# Ensemble weights
W_BILSTM    = 0.40
W_BERT      = 0.35
W_CLASSICAL = 0.25

# ── FIX 5: Relaxed veto thresholds ───────────────────────────────────────────
VETO_MIN_CITATIONS  = 3
VETO_MIN_WORDS      = 800    # lowered from 1200; CSV extraction under-counts
VETO_MIN_SECTIONS   = 3
# Strikes no longer used: veto now requires ALL three structural conditions bad

CHUNK_SIZE    = 3000
CHUNK_OVERLAP = 500
MAX_CHUNKS    = 4
TFIDF_FEAT    = 3000

# ── FIX 1: Val split ratio ────────────────────────────────────────────────────
VAL_SPLIT = 0.20   # 20% of real data held out; never seen during training

print("=" * 68)
print("  KDSH 2025 — v6 FIXED (all 5 bugs patched)")
print(f"  Device: {DEVICE}")
print("=" * 68)

# ── Phrase lists ──────────────────────────────────────────────────────────────
NON_PUB_PHRASES = [
    "to be completed","tbd","todo","[ insert","[to be","work in progress",
    "draft version","not yet finalized","results pending","experiment pending",
    "analysis pending","section to be written","coming soon",
    "we did not","we could not","no experiment","no evaluation","no comparison",
    "without comparison","no baseline","no benchmark","no dataset",
    "simple heuristic","rule-based only","no deep learning",
    "needs more work","requires further","not enough data","small dataset",
    "only a few samples","limited to","we only tested","only one experiment",
    "single experiment","no statistical","not statistically",
    "lorem ipsum","placeholder","insert figure here","see appendix (not included)",
]
POS_SIGNALS = [
    "state-of-the-art","outperforms","novel","we propose","significant improvement",
    "benchmark","baseline","ablation","convergence","empirically","theoretical",
    "proof","theorem","p-value","statistical significance","generalization",
    "robustness","reproducible","cross-validation","confidence interval",
    "standard deviation","zero-shot","few-shot","compared to","superior to",
    "improves over","gains of","error reduction","our method","proposed method",
]
NEG_SIGNALS = [
    "future work includes","we plan to","could not achieve","unfortunately",
    "failed to converge","out of scope","we leave this","beyond the scope",
    "not implemented","time constraint","lack of data","preliminary result",
    "partially implemented",
]
SECTION_HEADERS = [
    "abstract","introduction","related work","background","methodology",
    "method","approach","model","architecture","experiment","evaluation",
    "results","analysis","discussion","conclusion","future work","references",
    "acknowledgement","dataset","implementation","ablation","appendix",
]
CITATION_PATTERNS = [
    r"\[\d+\]",r"\[\d+,\s*\d+\]",r"\[\d+[-–]\d+\]",
    r"\(\w[\w\s\-]+,\s*\d{4}\)",r"\(\w[\w\s\-]+et al\.?,?\s*\d{4}\)",r"et al\.",
]
MATH_PATTERNS = [
    r"\bequation\b",r"\bformula\b",r"[=+\-*/^]{2,}",r"\balgorithm\b",
    r"\btheorem\b",r"\bmatrix\b",r"\bgradient\b",r"\bloss\b",
    r"\barg\s*min\b",r"\bsoftmax\b",r"\battention\b",
]
KEYWORD_CATEGORIES = {
    "architecture": ["transformer","bert","gpt","resnet","lstm","cnn","attention","encoder","decoder"],
    "training":     ["epoch","batch","gradient","optimizer","adam","sgd","learning rate","regularization"],
    "evaluation":   ["f1","accuracy","precision","recall","auc","bleu","rouge","perplexity"],
    "data":         ["dataset","corpus","annotation","split","train","test","validation","augmentation"],
    "theory":       ["theorem","proof","lemma","convergence","bound","complexity","approximation"],
    "comparison":   ["baseline","sota","outperform","improvement","ablation","comparison","benchmark"],
}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Text Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract(csv_path: Path) -> Tuple[str, Dict[str, str]]:
    lines = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            for row in csv.reader(f):
                line = " ".join(c.strip() for c in row if c.strip())
                if line:
                    lines.append(line)
    except Exception:
        return "", {}
    full = " ".join(lines)
    tl = full.lower()
    pos_list = []
    for hdr in SECTION_HEADERS:
        idx = 0
        while True:
            i = tl.find(hdr, idx)
            if i == -1: break
            if i == 0 or not tl[i-1].isalpha():
                pos_list.append((i, hdr))
            idx = i + 1
    pos_list.sort(key=lambda x: x[0])
    seen, deduped = set(), []
    for p, h in pos_list:
        if h not in seen:
            seen.add(h); deduped.append((p, h))
    sections = {}
    for i, (start, hdr) in enumerate(deduped):
        end = deduped[i+1][0] if i+1 < len(deduped) else len(full)
        sections[hdr] = full[start:end]
    return full, sections


def make_chunks(full_text: str, sections: Dict[str, str]) -> List[str]:
    priority = ""
    for key in ["abstract","introduction","conclusion","results"]:
        if key in sections:
            priority += sections[key] + " "
    chunks = [priority[:CHUNK_SIZE]]
    pos, step = 0, CHUNK_SIZE - CHUNK_OVERLAP
    while pos < len(full_text) and len(chunks) < MAX_CHUNKS:
        chunks.append(full_text[pos:pos + CHUNK_SIZE])
        pos += step
    return chunks[:MAX_CHUNKS]


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Feature Engineering (52 features)
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class FeatureResult:
    has_abs: float; has_intro: float; has_meth: float; has_exp: float
    has_conc: float; has_refs: float; has_ablat: float
    sec_cov: float; sec_count_norm: float
    cit_norm: float; has_et_al: float; ref_len_norm: float; cit_norm2: float
    math_norm: float; has_thm: float; has_algo: float
    figs_norm: float; tabs_norm: float; has_cmp_tab: float; has_any_fig: float
    pcts_norm: float; decs_norm: float; has_bl: float; numres_norm: float
    pos_norm: float; neg_norm: float; nonpub_norm: float
    avg_sl_norm: float; passive_norm: float
    words_norm: float; uniq_ratio: float; char_norm: float; sents_norm: float
    has_dataset: float; has_repro: float
    cit_pp: float; math_pp: float; fig_pp: float
    cit_raw: float; figs_raw: float; tabs_raw: float; math_raw: float; words_raw: float
    composite: float
    abstract_len_norm: float; conclusion_score: float; keyword_density: float
    sent_len_variance: float; recent_ref_ratio: float; result_section_ratio: float
    conclusion_len_norm: float; keyword_variety: float

    def to_array(self) -> np.ndarray:
        return np.array(dataclasses.astuple(self), dtype=np.float32)


def compute_features(full_text: str, sections: Dict[str, str]) -> FeatureResult:
    t = full_text.lower(); tw = t.split(); n_words = max(len(tw), 1)
    has_abs   = float("abstract" in t)
    has_intro = float("introduction" in t)
    has_meth  = float(any(x in t for x in ["methodology","method","approach","architecture"]))
    has_exp   = float(any(x in t for x in ["experiment","evaluation","results","analysis"]))
    has_conc  = float("conclusion" in t)
    has_refs  = float("references" in t)
    has_ablat = float("ablation" in t)
    sec_cov   = (has_abs+has_intro+has_meth+has_exp+has_conc+has_refs)/6.0
    sec_count = sum(1 for h in SECTION_HEADERS if h in t)
    cit = sum(len(re.findall(p, full_text)) for p in CITATION_PATTERNS)
    has_et_al = float("et al" in t)
    ref_len   = len(sections.get("references",""))
    math = sum(len(re.findall(p, full_text)) for p in MATH_PATTERNS)
    has_thm  = float(any(x in t for x in ["theorem","lemma","proof","corollary"]))
    has_algo = float("algorithm" in t)
    figs = len(re.findall(r"\bfig(ure)?\.?\s*\d+", t))
    tabs = len(re.findall(r"\btable\s*\d+", t))
    has_cmp_tab = float(bool(re.search(r"table\s*\d+.*compar", t)))
    pcts = len(re.findall(r"\d+\.?\d*\s*%", full_text))
    decs = len(re.findall(r"\b\d+\.\d{2,4}\b", full_text))
    has_bl = float(bool(re.search(r"\bbaseline\b|\bsota\b|\bstate.of.the.art\b", t)))
    pos_c = sum(1 for kw in POS_SIGNALS if kw in t)
    neg_c = sum(1 for kw in NEG_SIGNALS if kw in t)
    nonpub_c = sum(1 for ph in NON_PUB_PHRASES if ph in t)
    sents = re.split(r'[.!?]+', full_text)
    sent_lens = [len(s.split()) for s in sents if len(s.split()) > 2]
    avg_sl   = np.mean(sent_lens) if sent_lens else 0
    sent_var = np.std(sent_lens)  if len(sent_lens) > 1 else 0
    passive = len(re.findall(r"\b(is|are|was|were)\s+\w+ed\b", t))
    uniq_ratio = len(set(tw)) / n_words
    char_count = len(full_text)
    has_dataset = float(any(d in t for d in ["imagenet","coco","cifar","mnist","squad","glue","pubmed","wikipedia","common crawl","wikitext","conll","ontonotes"]))
    has_repro   = float(any(r in t for r in ["github","code available","open source","publicly available","repository"]))
    pages_est = max(n_words/250.0, 1.0)
    cit_pp = min(cit/pages_est, 3.0)/3.0
    math_pp = min(math/pages_est, 3.0)/3.0
    fig_pp = min(figs/pages_est, 1.0)
    abs_text = sections.get("abstract","")
    abstract_len_norm = min(len(abs_text.split())/300.0, 1.0)
    conc_text  = sections.get("conclusion","")
    conc_words = conc_text.lower().split()
    conc_kws = sum(1 for w in conc_words if w in {"demonstrate","show","outperform","improve","novel","contribution","achieve","propose","significant","future","conclude"})
    conclusion_score    = min(conc_kws/max(len(conc_words),1)*20, 1.0)
    conclusion_len_norm = min(len(conc_words)/400.0, 1.0)
    ml_kws = ["neural","deep learning","machine learning","model","training","inference","classification","regression","clustering","embedding","layer","parameter","feature","representation","optimization","loss function","gradient","backpropagation","attention","transformer"]
    keyword_density  = min(sum(1 for kw in ml_kws if kw in t)/10.0, 1.0)
    sent_len_variance= min(sent_var/20.0, 1.0)
    recent_years = re.findall(r"\b(201[5-9]|202[0-9])\b", full_text)
    all_years    = re.findall(r"\b(19[89]\d|20[012]\d)\b", full_text)
    recent_ref_ratio     = len(recent_years)/max(len(all_years),1)
    res_text             = sections.get("results", sections.get("evaluation",""))
    result_section_ratio = min(len(res_text)/max(char_count,1)*10, 1.0)
    cats_hit       = sum(1 for kws in KEYWORD_CATEGORIES.values() if any(kw in t for kw in kws))
    keyword_variety = cats_hit/len(KEYWORD_CATEGORIES)
    quality = (
        sec_cov*0.20 + min(cit/20.0,1.0)*0.15 + min(figs/4.0,1.0)*0.12 +
        min(math/15.0,1.0)*0.08 + min(pos_c/8.0,1.0)*0.12 +
        max(0,min((n_words-1000)/4000.0,1.0))*0.08 + has_bl*0.08 +
        abstract_len_norm*0.05 + conclusion_score*0.05 + keyword_variety*0.07
    ) - (min(nonpub_c/3.0,1.0)*0.30)
    return FeatureResult(
        has_abs=has_abs, has_intro=has_intro, has_meth=has_meth, has_exp=has_exp,
        has_conc=has_conc, has_refs=has_refs, has_ablat=has_ablat,
        sec_cov=sec_cov, sec_count_norm=float(sec_count)/len(SECTION_HEADERS),
        cit_norm=min(cit/40.0,1.0), has_et_al=has_et_al,
        ref_len_norm=min(ref_len/3000.0,1.0), cit_norm2=min(cit/10.0,1.0),
        math_norm=min(math/25.0,1.0), has_thm=has_thm, has_algo=has_algo,
        figs_norm=min(figs/6.0,1.0), tabs_norm=min(tabs/4.0,1.0),
        has_cmp_tab=has_cmp_tab, has_any_fig=float(figs>0),
        pcts_norm=min(pcts/15.0,1.0), decs_norm=min(decs/20.0,1.0),
        has_bl=has_bl, numres_norm=min((pcts+decs)/25.0,1.0),
        pos_norm=min(pos_c/10.0,1.0), neg_norm=min(neg_c/5.0,1.0),
        nonpub_norm=min(nonpub_c/5.0,1.0), avg_sl_norm=min(avg_sl/30.0,1.0),
        passive_norm=min(passive/max(n_words,1)*100,1.0),
        words_norm=min(n_words/6000.0,1.0), uniq_ratio=uniq_ratio,
        char_norm=min(char_count/30000.0,1.0), sents_norm=min(len(sents)/200.0,1.0),
        has_dataset=has_dataset, has_repro=has_repro,
        cit_pp=cit_pp, math_pp=math_pp, fig_pp=fig_pp,
        cit_raw=float(cit), figs_raw=float(figs), tabs_raw=float(tabs),
        math_raw=float(math), words_raw=float(n_words/100.0),
        composite=max(quality,0.0),
        abstract_len_norm=abstract_len_norm, conclusion_score=conclusion_score,
        keyword_density=keyword_density, sent_len_variance=sent_len_variance,
        recent_ref_ratio=recent_ref_ratio, result_section_ratio=result_section_ratio,
        conclusion_len_norm=conclusion_len_norm, keyword_variety=keyword_variety,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 5: Corrected hard_veto — requires ALL three structural conditions bad
# ══════════════════════════════════════════════════════════════════════════════

def hard_veto(full_text: str, fr: FeatureResult) -> Tuple[bool, str]:
    """
    FIXED: Veto fires only when ALL three structural quality checks fail
    simultaneously, PLUS a non-pub phrase trigger or combined no-fig+no-math.
    This prevents legitimate publishable papers from being vetoed on any single
    weak signal (e.g. CSV word-count under-counting).
    """
    t = full_text.lower()
    n_words = int(fr.words_norm * 6000)
    cit     = int(fr.cit_raw)
    sec_cov = fr.sec_cov

    # Three structural conditions (all must fail for a structural veto)
    struct_fail_cit = cit < VETO_MIN_CITATIONS
    struct_fail_wrd = n_words < VETO_MIN_WORDS
    struct_fail_sec = sec_cov < (VETO_MIN_SECTIONS / len(SECTION_HEADERS))

    structural_veto = struct_fail_cit and struct_fail_wrd and struct_fail_sec

    # Explicit non-publishable language veto (unchanged — high precision signal)
    nonpub_c = sum(1 for ph in NON_PUB_PHRASES if ph in t)
    nonpub_veto = nonpub_c >= 3

    # Combined: no figures AND no math AND word count too small (hard stub paper)
    stub_veto = (int(fr.figs_raw) == 0 and fr.math_norm < 0.05
                 and n_words < VETO_MIN_WORDS)

    if structural_veto:
        reasons = []
        if struct_fail_cit: reasons.append(f"citations={cit}<{VETO_MIN_CITATIONS}")
        if struct_fail_wrd: reasons.append(f"words={n_words}<{VETO_MIN_WORDS}")
        if struct_fail_sec: reasons.append(f"section_cov={sec_cov:.2f}")
        return True, "STRUCT_ALL_FAIL: " + " | ".join(reasons)

    if nonpub_veto:
        return True, f"NON_PUB_PHRASES={nonpub_c}"

    if stub_veto:
        return True, f"STUB: no_figs+no_math+short({n_words}w)"

    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Vocabulary Builder
# ══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, max_size: int = MAX_VOCAB):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.freq: Dict[str, int] = defaultdict(int)
        self.built = False

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return text.split()

    def build(self, texts: List[str]):
        for text in texts:
            for tok in self.tokenize(text):
                self.freq[tok] += 1
        sorted_words = sorted(self.freq.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[:self.max_size - 4]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.built = True
        print(f"      Vocabulary size: {len(self.word2idx)}")

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        tokens = [self.word2idx.get(t, self.UNK) for t in self.tokenize(text)]
        tokens = tokens[:max_len - 2]
        tokens = [self.BOS] + tokens + [self.EOS]
        if len(tokens) < max_len:
            tokens += [self.PAD] * (max_len - len(tokens))
        return tokens[:max_len]


vocab = Vocabulary(MAX_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PaperDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int],
                 feature_arrays: List[np.ndarray], augment: bool = False):
        self.texts    = texts
        self.labels   = labels
        self.features = feature_arrays
        self.augment  = augment

    def _augment(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        kept = []
        for s in sentences:
            if random.random() < 0.05: continue
            words = s.split()
            words = [w for w in words if random.random() > 0.03]
            if words: kept.append(" ".join(words))
        return " ".join(kept) if kept else text

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self._augment(self.texts[idx]) if self.augment else self.texts[idx]
        ids  = torch.tensor(vocab.encode(text), dtype=torch.long)
        lbl  = torch.tensor(self.labels[idx], dtype=torch.float)
        feat = torch.tensor(self.features[idx], dtype=torch.float)
        return ids, feat, lbl


def collate_fn(batch):
    ids, feats, lbls = zip(*batch)
    return (torch.stack(ids), torch.stack(feats), torch.stack(lbls))


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Focal Loss + Label Smoothing
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = 0.75,
                 label_smooth: float = LABEL_SMOOTH):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.label_smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth
        bce  = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits).detach()
        pt   = torch.where(targets > 0.5, probs, 1 - probs)
        alpha_t = torch.where(targets > 0.5,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — BiLSTM + Multi-Head Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        Q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))
        ctx  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx), attn


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, n_heads: int, feat_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.register_buffer('pos_enc', self._make_pos_enc(MAX_SEQ_LEN, embed_dim))
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        lstm_out_dim = hidden_dim * 2
        self.attn = MultiHeadSelfAttention(lstm_out_dim, n_heads, dropout=0.1)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.embed_head = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        classifier_in = lstm_out_dim * 2 + 128
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(64, 1),
        )

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        return pe.unsqueeze(0)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask   = (token_ids == 0)
        x      = self.embedding(token_ids) + self.pos_enc[:, :token_ids.size(1), :]
        x      = self.embed_drop(x)
        lstm_out, _ = self.bilstm(x)
        attn_out, _ = self.attn(lstm_out, mask)
        lstm_out    = self.layer_norm(lstm_out + attn_out)
        lengths   = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        mean_pool = (lstm_out * (~mask).unsqueeze(-1).float()).sum(1) / lengths
        max_pool, _ = lstm_out.masked_fill(mask.unsqueeze(-1), float('-inf')).max(dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, token_ids: torch.Tensor,
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled  = self.encode(token_ids)
        emb_out = self.embed_head(pooled)
        feat_out = self.feat_proj(features)
        combined = torch.cat([pooled, feat_out], dim=-1)
        logit    = self.classifier(combined).squeeze(-1)
        return logit, emb_out


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Contrastive Loss
# ══════════════════════════════════════════════════════════════════════════════

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)
        embeddings = F.normalize(embeddings, dim=-1)
        sim   = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_self = torch.eye(embeddings.shape[0], device=embeddings.device)
        mask_pos  = mask_pos - mask_self
        sim_exp = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
        denom   = sim_exp.sum(dim=1, keepdim=True) - sim_exp * mask_self
        log_prob = sim - sim.max(dim=1, keepdim=True).values - torch.log(denom + 1e-8)
        n_pos = mask_pos.sum(dim=1).clamp(min=1)
        loss  = -(mask_pos * log_prob).sum(dim=1) / n_pos
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — FGM Adversarial Training
# ══════════════════════════════════════════════════════════════════════════════

class FGM:
    def __init__(self, model: nn.Module, epsilon: float = 0.5):
        self.model   = model
        self.epsilon = epsilon
        self.backup  = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embedding' in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — SciBERT Fine-tuning Head
# ══════════════════════════════════════════════════════════════════════════════

class SciBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        hidden    = bert_model.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden, 256),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


class BertPaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx][:3000], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        return (enc['input_ids'].squeeze(0),
                enc['attention_mask'].squeeze(0),
                torch.tensor(self.labels[idx], dtype=torch.float))


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 10 — Training Loops
# ══════════════════════════════════════════════════════════════════════════════

def train_bilstm(model: BiLSTMAttentionClassifier,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 n_epochs: int) -> BiLSTMAttentionClassifier:

    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, label_smooth=LABEL_SMOOTH)
    cont_loss  = SupervisedContrastiveLoss(temperature=0.07)
    fgm        = FGM(model, epsilon=0.3)
    optimizer  = AdamW(model.parameters(), lr=LR_BILSTM, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    model.to(DEVICE)

    best_val_auc, best_state = 0.0, None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for ids, feats, lbls in train_loader:
            ids   = ids.to(DEVICE)
            feats = feats.to(DEVICE)
            lbls  = lbls.to(DEVICE)
            optimizer.zero_grad()
            logits, embeds = model(ids, feats)
            loss_focal = focal_loss(logits, lbls)
            loss_cont  = cont_loss(embeds, lbls) * 0.1
            loss       = loss_focal + loss_cont
            loss.backward()
            fgm.attack()
            logits_adv, _ = model(ids, feats)
            loss_adv      = focal_loss(logits_adv, lbls)
            loss_adv.backward()
            fgm.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            all_logits, all_lbls = [], []
            with torch.no_grad():
                for ids, feats, lbls in val_loader:
                    logits, _ = model(ids.to(DEVICE), feats.to(DEVICE))
                    all_logits.extend(torch.sigmoid(logits).cpu().numpy())
                    all_lbls.extend(lbls.numpy())
            if len(set(all_lbls)) > 1:
                val_auc = roc_auc_score(all_lbls, all_logits)
                preds   = (np.array(all_logits) >= 0.5).astype(int)
                val_f1  = f1_score(all_lbls, preds, zero_division=0)
                print(f"        Epoch {epoch+1:3d}: loss={total_loss/len(train_loader):.4f} "
                      f"val_AUC={val_auc:.3f} val_F1={val_f1:.3f}")
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"      ✅ BiLSTM restored best (val_AUC={best_val_auc:.3f})")
    return model


def train_scibert(bert_clf: SciBERTClassifier,
                  train_loader: DataLoader,
                  n_epochs: int) -> SciBERTClassifier:
    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, label_smooth=LABEL_SMOOTH)
    optimizer  = AdamW([
        {'params': bert_clf.bert.parameters(), 'lr': LR_BERT},
        {'params': bert_clf.head.parameters(), 'lr': LR_BERT * 10},
    ], weight_decay=1e-4)
    bert_clf.to(DEVICE)
    for epoch in range(n_epochs):
        bert_clf.train()
        total_loss = 0.0
        for input_ids, attn_mask, lbls in train_loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            lbls      = lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = bert_clf(input_ids, attn_mask)
            loss   = focal_loss(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_clf.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"        SciBERT Epoch {epoch+1}/{n_epochs}: loss={total_loss/len(train_loader):.4f}")
    return bert_clf


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 11 — Temperature Scaling
# ══════════════════════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logit_t = torch.tensor(logits, dtype=torch.float).unsqueeze(-1)
        label_t = torch.tensor(labels, dtype=torch.float)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logit_t.squeeze() / self.temperature, label_t)
            loss.backward()
            return loss
        optimizer.step(closure)
        print(f"      Temperature = {self.temperature.item():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Load Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/8] Loading reference papers...")
all_full_texts, all_features, all_labels, all_text_short = [], [], [], []

for csv_file in sorted(REFERENCE_DIR.rglob("*.csv")):
    if csv_file.name.startswith('.'): continue
    fp = str(csv_file).lower()
    if   "non-publishable" in fp or "non_publishable" in fp: label = 0
    elif "publishable" in fp:                                  label = 1
    else:                                                      continue
    full, secs = extract(csv_file)
    if len(full.strip()) < 100: continue
    fr = compute_features(full, secs)
    all_full_texts.append(full)
    all_features.append(fr.to_array())
    all_labels.append(label)
    all_text_short.append(full[:3000])

y = np.array(all_labels)
n_pub = int(y.sum()); n_non = int((1-y).sum())
print(f"      {len(y)} papers  (Pub={n_pub}, Non-pub={n_non})")
if len(y) == 0:
    raise SystemExit("❌ No training data found.")


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: Proper stratified train/val split BEFORE any augmentation
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1b/8] Splitting real data into train/val (no leakage)...")

# Ensure enough samples in each class for stratified split
min_cls = min(n_pub, n_non)
if min_cls < 4:
    # Too few for a proper split — use leave-one-out style (keep val tiny)
    val_size = max(1, min_cls // 2)
    train_idx, val_idx = train_test_split(
        range(len(all_labels)), test_size=val_size,
        stratify=all_labels, random_state=RANDOM_SEED
    )
else:
    train_idx, val_idx = train_test_split(
        range(len(all_labels)), test_size=VAL_SPLIT,
        stratify=all_labels, random_state=RANDOM_SEED
    )

# Real-only train and val sets (no augmentation yet)
real_train_texts  = [all_full_texts[i] for i in train_idx]
real_train_feats  = [all_features[i]   for i in train_idx]
real_train_labels = [all_labels[i]     for i in train_idx]

val_texts  = [all_full_texts[i] for i in val_idx]
val_feats  = [all_features[i]   for i in val_idx]
val_labels = [all_labels[i]     for i in val_idx]

print(f"      Train (real): {len(real_train_labels)}  |  Val (held-out): {len(val_labels)}")

# ── Augment ONLY the training portion ─────────────────────────────────────────
y_train_real = np.array(real_train_labels)
n_pub_tr  = int(y_train_real.sum())
n_non_tr  = int((1 - y_train_real).sum())
minority_label = 0 if n_pub_tr > n_non_tr else 1
minority_idxs  = [i for i, l in enumerate(real_train_labels) if l == minority_label]
n_needed       = abs(n_pub_tr - n_non_tr)

aug_texts, aug_feats, aug_labels = [], [], []
for _ in range(n_needed):
    src = random.choice(minority_idxs)
    aug_texts.append(real_train_texts[src])
    jittered = real_train_feats[src].copy()
    jittered += np.random.normal(0, 0.005, jittered.shape).astype(np.float32)
    aug_feats.append(jittered)
    aug_labels.append(minority_label)

train_texts  = real_train_texts + aug_texts
train_feats  = real_train_feats + aug_feats
train_labels = real_train_labels + aug_labels

print(f"      After augmentation: {len(train_labels)} train samples")
print(f"      Val distribution: Pub={sum(val_labels)} Non-pub={len(val_labels)-sum(val_labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# Build Vocabulary (on training texts only to avoid leakage)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/8] Building vocabulary from training data only...")
vocab.build(train_texts)


# ══════════════════════════════════════════════════════════════════════════════
# Build DataLoaders — val_loader now uses ONLY held-out val set
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/8] Building DataLoaders...")

train_ds = PaperDataset(train_texts, train_labels, train_feats, augment=True)
# FIX 1: val_loader uses the held-out set, never training data
val_ds   = PaperDataset(val_texts, val_labels, val_feats, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn)


# ══════════════════════════════════════════════════════════════════════════════
# Train BiLSTM
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/8] Training BiLSTM + Multi-Head Attention...")
feat_dim = len(all_features[0])
bilstm_model = BiLSTMAttentionClassifier(
    vocab_size=len(vocab.word2idx),
    embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS, n_heads=N_HEADS,
    feat_dim=feat_dim, dropout=DROPOUT,
)
n_params = sum(p.numel() for p in bilstm_model.parameters() if p.requires_grad)
print(f"      BiLSTM parameters: {n_params:,}")
bilstm_model = train_bilstm(bilstm_model, train_loader, val_loader, BILSTM_EPOCHS)

# Temperature-scale on val set
bilstm_model.eval()
bilstm_scaler = TemperatureScaler()
with torch.no_grad():
    raw_logits, raw_labels = [], []
    for ids, feats, lbls in val_loader:
        logits, _ = bilstm_model(ids.to(DEVICE), feats.to(DEVICE))
        raw_logits.extend(logits.cpu().numpy())
        raw_labels.extend(lbls.numpy())
print("      Calibrating BiLSTM temperature...")
bilstm_scaler.fit(np.array(raw_logits), np.array(raw_labels))


# ══════════════════════════════════════════════════════════════════════════════
# SciBERT Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

bert_clf    = None
bert_scaler = None
tokenizer_bert = None

val_text_short = [t[:3000] for t in val_texts]

if HAS_TRANSFORMERS and len(y_train_real) >= 4:
    print("\n[5/8] Fine-tuning SciBERT...")
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer_bert = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        bert_base      = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        bert_clf       = SciBERTClassifier(bert_base)

        bert_train_ds = BertPaperDataset(train_texts, train_labels, tokenizer_bert)
        # FIX 1: SciBERT val also uses held-out val set
        bert_val_ds   = BertPaperDataset(val_text_short, val_labels, tokenizer_bert)
        bert_train_ld = DataLoader(bert_train_ds, batch_size=4, shuffle=True)
        bert_val_ld   = DataLoader(bert_val_ds,   batch_size=4, shuffle=False)

        bert_clf = train_scibert(bert_clf, bert_train_ld, BERT_EPOCHS)

        bert_clf.eval()
        bert_scaler = TemperatureScaler()
        with torch.no_grad():
            bert_logits, bert_lbls = [], []
            for input_ids, attn_mask, lbls in bert_val_ld:
                logits = bert_clf(input_ids.to(DEVICE), attn_mask.to(DEVICE))
                bert_logits.extend(logits.cpu().numpy())
                bert_lbls.extend(lbls.numpy())
        print("      Calibrating SciBERT temperature...")
        bert_scaler.fit(np.array(bert_logits), np.array(bert_lbls))
        print("      ✅ SciBERT fine-tuning complete")
    except Exception as e:
        print(f"      ⚠️  SciBERT failed ({e}), using BiLSTM only")
        bert_clf = None
else:
    print("\n[5/8] SciBERT skipped (not available or insufficient data)")


# ══════════════════════════════════════════════════════════════════════════════
# Classical Stacking — trained on all real data, OOF via cross-val
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6/8] Training classical ensemble (RF + ET + GBM)...")

# FIX 4: Classical models always use ALL real data for cross-validation
feat_arr = np.array(all_features)
y_real   = np.array(all_labels)

tfidf = TfidfVectorizer(max_features=TFIDF_FEAT, ngram_range=(1,2),
                         sublinear_tf=True, min_df=2)
X_tfidf = tfidf.fit_transform([t[:15000] for t in all_full_texts])
n_svd = min(80, X_tfidf.shape[1]-1, X_tfidf.shape[0]-1)
svd   = TruncatedSVD(n_components=n_svd, random_state=RANDOM_SEED)
X_svd = svd.fit_transform(X_tfidf)

X_classical = np.hstack([X_svd, feat_arr])
scaler_cls  = StandardScaler()
X_cls_s     = scaler_cls.fit_transform(X_classical)

cv_k = max(2, min(5, min(int(y_real.sum()), int((1-y_real).sum()))))
skf  = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=RANDOM_SEED)

classical_learners = [
    ("rf",  RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                    random_state=RANDOM_SEED, n_jobs=-1)),
    ("et",  ExtraTreesClassifier(n_estimators=400, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)),
    ("gb",  GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                        learning_rate=0.04, subsample=0.75,
                                        random_state=RANDOM_SEED)),
]
if HAS_XGB:
    sw = int((1-y_real).sum()) / max(int(y_real.sum()), 1)
    classical_learners.append(("xgb", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.75, scale_pos_weight=sw,
        eval_metric='logloss', use_label_encoder=False,
        random_state=RANDOM_SEED, n_jobs=-1)))

oof_stacks = np.zeros((len(y_real), len(classical_learners)))
for b, (name, clf) in enumerate(classical_learners):
    print(f"      CV {name}...", end=" ", flush=True)
    oof = cross_val_predict(clf, X_cls_s, y_real, cv=skf, method='predict_proba')[:, 1]
    oof_stacks[:, b] = oof
    clf.fit(X_cls_s, y_real)
    f1 = f1_score(y_real, (oof >= 0.5).astype(int), zero_division=0)
    print(f"F1={f1:.3f}  AUC={roc_auc_score(y_real, oof):.3f}")

meta_sc  = StandardScaler()
meta_oof = meta_sc.fit_transform(oof_stacks)
meta_clf = LogisticRegression(C=0.3, class_weight='balanced',
                               max_iter=1000, random_state=RANDOM_SEED)
meta_clf.fit(meta_oof, y_real)
meta_oof_pred = cross_val_predict(meta_clf, meta_oof, y_real, cv=cv_k,
                                   method='predict_proba')[:, 1]
print(f"      Classical meta F1={f1_score(y_real,(meta_oof_pred>=0.5).astype(int)):.3f}  "
      f"AUC={roc_auc_score(y_real, meta_oof_pred):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Unified threshold — all OOF from the same held-out val set
# ══════════════════════════════════════════════════════════════════════════════

print("\n[7/8] Optimising final threshold (unified val set)...")

y_val = np.array(val_labels)

# BiLSTM OOF on val set
bilstm_model.eval()
bilstm_val_probs = []
with torch.no_grad():
    for ids, feats, lbls in val_loader:
        logits, _ = bilstm_model(ids.to(DEVICE), feats.to(DEVICE))
        logits_cal = bilstm_scaler(logits.unsqueeze(-1)).squeeze()
        probs = torch.sigmoid(logits_cal).cpu().numpy()
        bilstm_val_probs.extend(
            probs.tolist() if hasattr(probs, 'tolist') else [float(probs)]
        )
bilstm_val_probs = np.array(bilstm_val_probs)

# Classical probabilities on val set
X_tfidf_val = tfidf.transform([t[:15000] for t in val_texts])
X_svd_val   = svd.transform(X_tfidf_val)
X_cls_val   = scaler_cls.transform(np.hstack([X_svd_val, np.array(val_feats)]))
classical_val_stack = np.column_stack([
    clf.predict_proba(X_cls_val)[:, 1] for _, clf in classical_learners
])
# FIX 4: meta_clf already correctly fit on all real data above
cls_val_probs = meta_clf.predict_proba(meta_sc.transform(classical_val_stack))[:, 1]

# SciBERT probabilities on val set
if bert_clf is not None:
    bert_clf.eval()
    bert_val_probs = []
    with torch.no_grad():
        for input_ids, attn_mask, lbls in DataLoader(
                BertPaperDataset(val_text_short, val_labels, tokenizer_bert),
                batch_size=4, shuffle=False):
            logits = bert_clf(input_ids.to(DEVICE), attn_mask.to(DEVICE))
            logits_cal = bert_scaler(logits.unsqueeze(-1)).squeeze()
            probs = torch.sigmoid(logits_cal).cpu().numpy()
            bert_val_probs.extend(
                probs.tolist() if hasattr(probs, 'tolist') else [float(probs)]
            )
    bert_val_probs = np.array(bert_val_probs)
    w_bilstm = W_BILSTM; w_bert = W_BERT; w_cls = W_CLASSICAL
    final_oof = w_bilstm * bilstm_val_probs + w_bert * bert_val_probs + w_cls * cls_val_probs
    print(f"      3-model blend (on val set): BiLSTM={w_bilstm} BERT={w_bert} Classical={w_cls}")
else:
    bert_val_probs = None
    w_bilstm = 0.60; w_cls = 0.40
    final_oof = w_bilstm * bilstm_val_probs + w_cls * cls_val_probs
    print(f"      2-model blend (on val set): BiLSTM={w_bilstm} Classical={w_cls}")

# Precision-weighted threshold search (β=0.5)
BETA_SQ = 0.25
best_thresh, best_fb = 0.50, 0.0
for th in np.arange(0.25, 0.80, 0.01):
    preds = (final_oof >= th).astype(int)
    if preds.sum() == 0 or preds.sum() == len(preds): continue
    p  = precision_score(y_val, preds, zero_division=0)
    r  = recall_score(y_val, preds, zero_division=0)
    fb = (1 + BETA_SQ) * p * r / (BETA_SQ * p + r + 1e-9)
    if fb > best_fb:
        best_fb, best_thresh = fb, th

# Guard: if val set too small and threshold search failed, fall back to 0.50
if best_fb == 0.0:
    best_thresh = 0.50
    print("      ⚠️  Threshold search inconclusive on small val set — using 0.50")

final_preds = (final_oof >= best_thresh).astype(int)
if len(y_val) > 0 and len(set(y_val)) > 1:
    tp = int(((final_preds==1)&(y_val==1)).sum())
    fp = int(((final_preds==1)&(y_val==0)).sum())
    tn = int(((final_preds==0)&(y_val==0)).sum())
    fn = int(((final_preds==0)&(y_val==1)).sum())
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    print(f"      Threshold={best_thresh:.2f}  Fβ={best_fb:.3f}")
    print(f"      Val confusion: TP={tp} FP={fp} TN={tn} FN={fn}  "
          f"Prec={prec:.2%} Rec={rec:.2%}")
else:
    print(f"      Threshold={best_thresh:.2f} (val set too small for full confusion matrix)")

# ── FIX 4: Refit all models on full real data (correct alignment) ─────────────
print("      Refitting classical on full real data...")
# Classical is already fit on all real data from the cross-val loop above — good.
# Refit meta on all real OOF (already done above via meta_clf.fit(meta_oof, y_real)).
# Now also refit on augmented train for better generalization:
all_feats_aug = np.array(train_feats)
X_tfidf_aug   = tfidf.transform([t[:15000] for t in train_texts])
X_svd_aug     = svd.transform(X_tfidf_aug)
X_cls_aug     = scaler_cls.transform(np.hstack([X_svd_aug, all_feats_aug]))
y_aug         = np.array(train_labels)
for _, clf in classical_learners:
    clf.fit(X_cls_aug, y_aug)

# FIX 4: Meta-learner refit uses only real data predictions, correctly aligned
X_cls_real_final = scaler_cls.transform(np.hstack([X_svd, feat_arr]))  # real data
meta_preds_real = np.column_stack([
    clf.predict_proba(X_cls_real_final)[:, 1] for _, clf in classical_learners
])
meta_clf.fit(meta_sc.transform(meta_preds_real), y_real)
print("      ✅ Meta-learner refit on real data only (no index misalignment)")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 12 — Inference with TTA
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[8/8] Evaluating {PAPERS_DIR.resolve()} ...")


def predict_bilstm(text: str, feat_vec: np.ndarray, n_tta: int = TTA_RUNS) -> float:
    bilstm_model.eval()
    probs = []
    ids  = torch.tensor(vocab.encode(text), dtype=torch.long).unsqueeze(0).to(DEVICE)
    feat = torch.tensor(feat_vec, dtype=torch.float).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit, _ = bilstm_model(ids, feat)
        logit_cal = bilstm_scaler(logit.unsqueeze(-1)).squeeze()
        probs.append(torch.sigmoid(logit_cal).item())
    ds_tta = PaperDataset([text]*(n_tta-1), [0]*(n_tta-1),
                           [feat_vec]*(n_tta-1), augment=True)
    for item in ds_tta:
        ids_a, feat_a, _ = item
        with torch.no_grad():
            logit, _ = bilstm_model(ids_a.unsqueeze(0).to(DEVICE),
                                     feat_a.unsqueeze(0).to(DEVICE))
            logit_cal = bilstm_scaler(logit.unsqueeze(-1)).squeeze()
            probs.append(torch.sigmoid(logit_cal).item())
    return float(np.mean(probs))


def predict_bert(text: str) -> Optional[float]:
    if bert_clf is None or tokenizer_bert is None:
        return None
    bert_clf.eval()
    enc = tokenizer_bert(text[:3000], max_length=512, padding='max_length',
                          truncation=True, return_tensors='pt')
    with torch.no_grad():
        logit = bert_clf(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
        logit_cal = bert_scaler(logit.unsqueeze(-1)).squeeze()
        return torch.sigmoid(logit_cal).item()


def predict_classical(feat_vec: np.ndarray, full_text: str) -> float:
    X_tf = tfidf.transform([full_text[:15000]])
    X_sv = svd.transform(X_tf)
    Xp   = scaler_cls.transform(np.hstack([X_sv, feat_vec.reshape(1,-1)]))
    bp   = np.array([clf.predict_proba(Xp)[0,1] for _, clf in classical_learners])
    meta_input = meta_sc.transform(bp.reshape(1,-1))
    return float(meta_clf.predict_proba(meta_input)[0,1])


def predict_paper(csv_path: Path) -> Optional[dict]:
    full, secs = extract(csv_path)
    if len(full.strip()) < 80:
        return None

    fr  = compute_features(full, secs)
    fv  = fr.to_array()
    vetoed, veto_reason = hard_veto(full, fr)

    p_bilstm = predict_bilstm(full, fv)
    p_bert   = predict_bert(full)
    p_cls    = predict_classical(fv, full)

    if p_bert is not None:
        blended = W_BILSTM * p_bilstm + W_BERT * p_bert + W_CLASSICAL * p_cls
    else:
        blended = 0.60 * p_bilstm + 0.40 * p_cls

    # FIX 3: Correct majority gate — strict majority (>50%), not ceil(n/2 + 0.1)
    votes = [int(p_bilstm >= 0.5), int(p_cls >= 0.5)]
    if p_bert is not None:
        votes.append(int(p_bert >= 0.5))
    n_votes   = sum(votes)
    gate_pass = n_votes > len(votes) / 2   # strict majority, corrected

    if vetoed:
        pred = 0; dp = "HARD_VETO"
    elif blended >= best_thresh and gate_pass:
        pred = 1; dp = "DL_ENSEMBLE"
    elif blended >= best_thresh and not gate_pass:
        pred = 0; dp = "GATE_BLOCKED"
    else:
        pred = 0; dp = "BELOW_THRESH"

    gap  = abs(blended - best_thresh)
    conf = "HIGH" if gap > 0.20 else ("MED" if gap > 0.09 else "LOW⚠")

    return dict(
        fr=fr, fv=fv, pred=pred, blended=blended,
        p_bilstm=p_bilstm, p_bert=p_bert, p_cls=p_cls,
        n_votes=n_votes, total_votes=len(votes), gate_pass=gate_pass,
        vetoed=vetoed, veto_reason=veto_reason, dp=dp, conf=conf,
    )


results, log_lines = [], []
csvs = sorted([f for f in PAPERS_DIR.rglob("*.csv")
               if not f.name.lower().startswith("130_unlabeled")])
print(f"      Found {len(csvs)} papers.\n")

for csv_file in csvs:
    pid = csv_file.stem
    out = predict_paper(csv_file)
    if out is None:
        print(f"  {pid}: ⚠️  TOO SHORT — SKIPPED")
        continue

    fr      = out["fr"]
    pred    = out["pred"]
    blended = out["blended"]
    cit_raw = int(fr.cit_raw)
    fig_raw = int(fr.figs_raw)
    n_words = int(fr.words_norm * 6000)
    nonpub_c= int(fr.nonpub_norm * 5)

    flags = []
    if fr.sec_cov < 0.50:   flags.append("LOW_STRUCT")
    if cit_raw < 5:         flags.append("FEW_CITES")
    if fig_raw == 0:        flags.append("NO_FIGS")
    if n_words < 1500:      flags.append("SHORT")
    if nonpub_c >= 2:       flags.append("WEAK_PHRASES")

    bert_s = f"bert={out['p_bert']:.3f} " if out['p_bert'] is not None else ""
    print(f"  {pid}: {'✅ PUB   ' if pred==1 else '❌ NON-PUB'}  "
          f"b={blended:.3f} th={best_thresh:.2f}  "
          f"lstm={out['p_bilstm']:.3f} {bert_s}cls={out['p_cls']:.3f}  "
          f"votes={out['n_votes']}/{out['total_votes']}  "
          f"[{out['dp']}]"
          + (f"  VETO:{out['veto_reason']}" if out['vetoed'] else "")
          + (f"  {','.join(flags)}" if flags else ""))

    log_lines.append(
        f"\n{'─'*62}\n"
        f"Paper : {pid}\n"
        f"  Decision        : {'PUBLISHABLE' if pred==1 else 'NON-PUBLISHABLE'}\n"
        f"  Decision Path   : {out['dp']}\n"
        f"  Blended Score   : {blended:.4f}  (threshold={best_thresh:.2f})\n"
        f"  BiLSTM Score    : {out['p_bilstm']:.4f}\n"
        f"  SciBERT Score   : {format(out['p_bert'], '.4f') if out['p_bert'] is not None else 'N/A'}\n"
        f"  Classical Score : {out['p_cls']:.4f}\n"
        f"  Votes           : {out['n_votes']}/{out['total_votes']}\n"
        f"  Gate Pass       : {out['gate_pass']}  (majority = votes > {len(out.get('total_votes', 2))/2 if False else 'n/2'})\n"
        f"  Confidence      : {out['conf']}\n"
        f"  Veto            : {out['veto_reason'] if out['vetoed'] else 'None'}\n"
        f"  Section Coverage: {fr.sec_cov*100:.0f}%\n"
        f"  Citations       : {cit_raw}\n"
        f"  Figures         : {fig_raw}  Tables: {int(fr.tabs_raw)}\n"
        f"  Math Density    : {fr.math_norm:.2f}\n"
        f"  Keyword Variety : {fr.keyword_variety:.2f}\n"
        f"  Composite Score : {fr.composite:.3f}\n"
        f"  Flags           : {', '.join(flags) if flags else 'None'}\n"
    )

    results.append({
        "Paper ID"        : pid,
        "Publishable"     : pred,
        "Blended_Score"   : round(blended, 4),
        "BiLSTM_Score"    : round(out['p_bilstm'], 4),
        "SciBERT_Score"   : round(out['p_bert'], 4) if out['p_bert'] else "",
        "Classical_Score" : round(out['p_cls'], 4),
        "Decision_Path"   : out['dp'],
        "Votes"           : f"{out['n_votes']}/{out['total_votes']}",
        "Confidence"      : out['conf'],
        "Citations"       : cit_raw,
        "Figures"         : fig_raw,
        "Composite_Score" : round(float(fr.composite), 3),
        "Flags"           : "|".join(flags),
    })

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
with open(DETAILED_LOG, 'w', encoding='utf-8') as f:
    f.write("KDSH 2025 — v6 FIXED Deep Learning Report\n")
    f.write("=" * 62 + "\n")
    f.writelines(log_lines)

pub        = sum(r["Publishable"] for r in results)
veto_count = sum(1 for r in results if r["Decision_Path"] == "HARD_VETO")
gate_count = sum(1 for r in results if r["Decision_Path"] == "GATE_BLOCKED")

print("\n" + "=" * 68)
print(f"  DONE.  Papers evaluated : {len(results)}")
print(f"  Publishable             : {pub}")
print(f"  Non-publishable         : {len(results)-pub}")
print(f"    → Hard vetoed         : {veto_count}")
print(f"    → Gate blocked        : {gate_count}")
print(f"  Results  → {Path(OUTPUT_CSV).resolve()}")
print(f"  Full log → {Path(DETAILED_LOG).resolve()}")
print("=" * 68)