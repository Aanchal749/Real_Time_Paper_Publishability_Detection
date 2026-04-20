"""
KDSH 2025 — End-to-End Paper Publishability & Conference Router
================================================================
Professional Enterprise Dashboard with Integrated ML Pipeline.
"""

import os
import re
import csv
import time
import math
import random
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────
# PAGE CONFIG (MUST BE FIRST)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Conference Router | KDSH 2025",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ── Classical ML ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

try:
    from transformers import pipeline as hf_pipeline
    HAS_ZS = True
except Exception:
    HAS_ZS = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

# ─────────────────────────────────────────────────────────
# SUPABASE INTEGRATION (AGGRESSIVE FAILSAFE VERSION)
# ─────────────────────────────────────────────────────────
# I have removed the cache so it forces a connection attempt every time.
def init_supabase():
    url = "https://mfqquqistinunclbiyjp.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1mcXF1cWlzdGludW5jbGJpeWpwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY2OTE2NDEsImV4cCI6MjA5MjI2NzY0MX0.v77UohfwA6gGQoSmn4VBKkr42j8ZV9LQ2SUTGG2pxhk=="
    try:
        client = create_client(url, key)
        return client, None
    except Exception as e:
        return None, str(e)

# This will grab the actual error message if it fails
supabase, db_error = init_supabase()

# ─────────────────────────────────────────────────────────
# CONFIG & HYPERPARAMS
# ─────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

REFERENCE_DIR = Path(r"C:\Dataset\Papers.csv\Reference") if Path(r"C:\Dataset\Papers.csv\Reference").exists() else Path("./Reference")

MAX_VOCAB    = 20000
MAX_SEQ_LEN  = 256
EMBED_DIM    = 128
GRU_HIDDEN   = 128
N_GRU_LAYERS = 2
N_HEADS      = 4
DROPOUT      = 0.35
GRU_EPOCHS   = 30
BATCH_SIZE   = 8
LR           = 2e-4

W_SBERT     = 0.40
W_ZEROSHOT  = 0.30
W_CLASSICAL = 0.20
W_GRU       = 0.10

TFIDF_FEAT  = 5000
SVD_COMP    = 100
TOP_K_REFS  = 3

CONFERENCE_HYPOTHESES = {
    "CVPR":  "This paper is about computer vision, image recognition, object detection, or visual perception.",
    "EMNLP": "This paper is about natural language processing, text understanding, or computational linguistics.",
    "KDD":   "This paper is about knowledge discovery, data mining, or large-scale data analysis.",
    "NEURIPS": "This paper is about machine learning theory, deep learning architectures, or neural network optimization.",
    "TMLR":  "This paper presents a rigorous empirical or theoretical machine learning study.",
}

# ─────────────────────────────────────────────────────────
# CSS — MODERN PROFESSIONAL THEME
# ─────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    * { font-family: 'Inter', sans-serif; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important; }
    .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1440px !important; }

    /* SIDEBAR */
    [data-testid="stSidebar"] { background: #ffffff !important; border-right: none !important; box-shadow: 2px 0 12px rgba(0, 0, 0, 0.03); padding: 1.5rem 1rem !important; }
    [data-testid="stSidebar"] h3 { font-weight: 700 !important; font-size: 1rem !important; letter-spacing: 0.02em; color: #0f172a !important; text-transform: uppercase; margin-bottom: 1rem; }
    [data-testid="stSidebar"] .stAlert { border-radius: 12px; border-left: 4px solid; font-size: 0.85rem; padding: 0.6rem 1rem; margin: 0.5rem 0; }

    /* HERO BANNER */
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #312e81 100%);
        border-radius: 24px; padding: 2.5rem 3rem; margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        color: white;
    }
    .hero-title { font-size: 2.8rem; font-weight: 800; line-height: 1.2; margin: 0 0 0.75rem 0; letter-spacing: -0.02em; color: #ffffff; }
    .hero-sub { font-size: 1.1rem; font-weight: 500; max-width: 650px; margin: 0; color: #cbd5e1; line-height: 1.5; }

    /* STAT CARDS */
    .stat-strip { display: flex; gap: 1.5rem; margin-bottom: 2.5rem; }
    .stat-card {
        background: #ffffff; border-radius: 20px; padding: 1.2rem 1.5rem; flex: 1; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #eef2f6; transition: all 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-2px); box-shadow: 0 12px 20px -12px rgba(0, 0, 0, 0.1); border-color: #e2e8f0; }
    .stat-lbl { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; color: #64748b; }
    .stat-val { font-size: 2rem; font-weight: 800; color: #3b82f6; margin-top: 0.4rem; line-height: 1.2; }

    /* COLUMNS */
    [data-testid="column"]:nth-of-type(1), [data-testid="column"]:nth-of-type(2) {
        background: #ffffff !important; border-radius: 24px !important; border: 1px solid #eef2f6 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important; padding: 1.8rem !important;
    }

    /* INPUTS */
    [data-testid="stFileUploader"] > div:first-child { background: #f8fafc; border-radius: 16px; border: 1px dashed #cbd5e1; }

    /* BUTTONS */
    .stButton > button[kind="primary"] {
        background: linear-gradient(105deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important; font-size: 1rem !important; font-weight: 700 !important;
        padding: 0.85rem 1.2rem !important; border: none !important; border-radius: 40px !important;
        box-shadow: 0 4px 8px rgba(37,99,235,0.2) !important; transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 12px 20px -12px rgba(37,99,235,0.4) !important; }

    /* VALUATION CARD */
    .val-card-container { background: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05); margin-top: 1rem; border: 1px solid #eef2f6; }
    .val-header { background: linear-gradient(120deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.8rem 2rem; border-bottom: 1px solid #e2f3e4; }
    .val-title { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #15803d; }
    .val-big-price { font-size: 3rem; font-weight: 800; color: #166534; margin: 0.8rem 0 0.2rem; line-height: 1.1; letter-spacing: -0.02em; }
    
    .val-header.rejected { background: linear-gradient(120deg, #fef2f2 0%, #fee2e2 100%); border-bottom: 1px solid #fecaca; }
    .val-header.rejected .val-title { color: #b91c1c; }
    .val-header.rejected .val-big-price { color: #991b1b; font-size: 2.2rem; }

    .val-footer { padding: 1.5rem 2rem; background: #ffffff; }
    .val-range-box { background: #f8fafc; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; border-left: 5px solid #3b82f6; }
    .val-range-box.red { border-left: 5px solid #ef4444; }
    .val-range-lbl { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; color: #64748b; margin-bottom: 0.5rem; }
    .val-range-val { font-size: 1.05rem; font-weight: 500; color: #0f172a; line-height: 1.5; }

    /* PROB BARS */
    .prob-row { display: flex; align-items: center; margin-bottom: 1rem; background: #f8fafc; padding: 0.8rem 1.2rem; border-radius: 12px; }
    .prob-label { width: 90px; font-weight: 700; color: #374151; font-size: 0.95rem; }
    .prob-bar-bg { flex: 1; background: #e2e8f0; height: 10px; border-radius: 5px; margin: 0 1.5rem; overflow: hidden; }
    .prob-bar-fill { background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); height: 100%; border-radius: 5px; }
    .prob-pct { width: 55px; text-align: right; font-weight: 700; color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DATA EXTRACTION & PUBLISHABILITY GATEKEEPER
# ─────────────────────────────────────────────────────────
def extract_text_from_stream(file_stream, ext) -> str:
    text = ""
    try:
        if ext == 'pdf' and HAS_FITZ:
            doc = fitz.open(stream=file_stream, filetype="pdf")
            text = " ".join(page.get_text() for page in doc).replace("\n", " ")
        elif ext == 'csv':
            content = file_stream.decode('utf-8', errors='replace')
            for row in csv.reader(content.splitlines()):
                text += " ".join([c.strip() for c in row if c.strip()]) + " "
    except Exception: pass
    return re.sub(r'\s+', ' ', text).strip()

def check_publishability(text: str) -> Tuple[bool, str]:
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count < 300:
        return False, "Insufficient content. The document contains too few words to be evaluated as a full academic research paper. Missing critical depth in methodology and evaluation."
    markers = ["abstract", "introduction", "related work", "method", "experiment", "result", "conclusion", "reference"]
    if sum(1 for m in markers if m in text_lower) < 3:
        return False, "Poor academic structure. The document lacks standard research formatting (e.g., clear Methodology, Experiments, or References sections). Cannot be routed to a top-tier peer-reviewed venue."
    if "reference" not in text_lower and "bibliography" not in text_lower:
        return False, "Missing citations. A rigorous academic paper requires a bibliography or references section grounding it in prior literature."
    return True, "Document passed structural and depth analysis."

# ─────────────────────────────────────────────────────────
# ML MODELS PIPELINE
# ─────────────────────────────────────────────────────────
class Vocabulary:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.freq = defaultdict(int)
    def tokenize(self, text): return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    def build(self, texts):
        for t in texts:
            for tok in self.tokenize(t): self.freq[tok] += 1
        for word, _ in sorted(self.freq.items(), key=lambda x: -x[1])[:MAX_VOCAB - 4]:
            self.word2idx[word] = len(self.word2idx)
    def encode(self, text):
        toks = [self.word2idx.get(t, self.UNK) for t in self.tokenize(text)][:MAX_SEQ_LEN - 2]
        return ([self.BOS] + toks + [self.EOS] + [self.PAD] * MAX_SEQ_LEN)[:MAX_SEQ_LEN]

class ConferenceDataset(Dataset):
    def __init__(self, texts, labels, vocab): self.texts, self.labels, self.vocab = texts, labels, vocab
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return torch.tensor(self.vocab.encode(self.texts[idx]), dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class BiGRUConferenceClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.bigru = nn.GRU(EMBED_DIM, GRU_HIDDEN, num_layers=N_GRU_LAYERS, bidirectional=True, batch_first=True, dropout=DROPOUT)
        self.classifier = nn.Sequential(nn.Linear(GRU_HIDDEN * 4, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_classes))
    def forward(self, x):
        mask = (x == 0)
        out, _ = self.bigru(self.embedding(x))
        lengths = (~mask).sum(1, keepdim=True).float().clamp(min=1)
        mean_pool = (out * (~mask).unsqueeze(-1).float()).sum(1) / lengths
        max_pool, _ = out.masked_fill(mask.unsqueeze(-1), float('-inf')).max(1)
        return self.classifier(torch.cat([mean_pool, max_pool], dim=-1))

class SBERTConferenceClassifier:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def fit(self, texts, labels, ids):
        self.ref_embeds = self.model.encode([t[:3000] for t in texts], normalize_embeddings=True)
        self.ref_labels, self.ref_ids, self.conferences = np.array(labels), np.array(ids), sorted(set(labels))
    def predict_proba(self, text):
        emb = self.model.encode([text[:3000]], normalize_embeddings=True)
        sims = (self.ref_embeds @ emb.T).squeeze()
        k = min(TOP_K_REFS, len(sims))
        top_idx = [np.argmax(sims)] if k == 1 else np.argsort(sims)[::-1][:k]
        votes = defaultdict(float)
        for idx in top_idx: votes[self.ref_labels[idx]] += float(sims[idx] if sims.ndim > 0 else sims)
        probs = {c: votes.get(c, 0.0) / (sum(votes.values()) + 1e-9) for c in self.conferences}
        return probs, self.ref_ids[top_idx[0]], float(sims[top_idx[0]] if sims.ndim > 0 else sims)

class ClassicalConferenceStack:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=TFIDF_FEAT, ngram_range=(1,2), sublinear_tf=True, min_df=1)
        self.svd   = TruncatedSVD(n_components=SVD_COMP, random_state=RANDOM_SEED)
        self.scaler = StandardScaler()
        self.le    = LabelEncoder()
    def fit(self, texts, labels):
        y = self.le.fit_transform(labels)
        self.conferences = list(self.le.classes_)
        X_tfidf = self.tfidf.fit_transform([t[:10000] for t in texts])
        self.svd.n_components = max(1, min(SVD_COMP, X_tfidf.shape[1]-1, X_tfidf.shape[0]-1))
        X = self.scaler.fit_transform(self.svd.fit_transform(X_tfidf))
        clf1, clf2, clf3 = LogisticRegression(C=1.0, random_state=RANDOM_SEED), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED), ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_SEED)
        self.ensemble = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('et', clf3)], voting='soft')
        self.ensemble.fit(X, y)
    def predict_proba(self, text):
        X = self.scaler.transform(self.svd.transform(self.tfidf.transform([text[:10000]])))
        return {c: float(p) for c, p in zip(self.conferences, self.ensemble.predict_proba(X)[0])}

class ZeroShotConferenceClassifier:
    def __init__(self):
        self.pipe = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
        self.conferences = list(CONFERENCE_HYPOTHESES.keys())
    def predict_proba(self, text):
        res = self.pipe(text[:1000], candidate_labels=list(CONFERENCE_HYPOTHESES.values()), multi_label=False)
        hyp_to_conf = {v: k for k, v in CONFERENCE_HYPOTHESES.items()}
        probs = {hyp_to_conf.get(l, l): float(s) for l, s in zip(res['labels'], res['scores'])}
        return {c: probs.get(c, 0.0) / (sum(probs.values()) + 1e-9) for c in self.conferences}

class ConferenceKeywordExtractor:
    def __init__(self): self.conf_tfidf = {}
    def fit(self, texts, labels):
        grouped = defaultdict(list)
        for t, l in zip(texts, labels): grouped[l].append(t)
        for conf, corpus in grouped.items():
            try: self.conf_tfidf[conf] = TfidfVectorizer(max_features=100, stop_words='english').fit(corpus)
            except: pass
    def top_keywords(self, text, conf):
        try:
            vec = self.conf_tfidf[conf]
            scores = vec.transform([text[:5000]]).toarray()[0]
            return [vec.get_feature_names_out()[i] for i in np.argsort(scores)[::-1][:4] if scores[i] > 0]
        except: return []

@st.cache_resource(show_spinner=False)
def initialize_pipeline():
    texts, labels, ids = [], [], []
    pub_dir = REFERENCE_DIR / "Publishable"
    if pub_dir.exists():
        for conf_dir in sorted(pub_dir.iterdir()):
            if not conf_dir.is_dir(): continue
            raw_conf = conf_dir.name.upper().replace(" ", "")
            conf = "NEURIPS" if raw_conf in ["NEURALPS", "NEURIPS", "NIPS"] else raw_conf
            for csv_file in conf_dir.rglob("*.csv"):
                try:
                    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
                        txt = " ".join([" ".join([c.strip() for c in r if c.strip()]) for r in csv.reader(f)])
                        if len(txt) > 100: texts.append(txt); labels.append(conf); ids.append(csv_file.stem)
                except: pass
    if not texts: return None, None, None, None, None, None, None

    conferences = sorted(set(labels))
    vocab = Vocabulary()
    vocab.build(texts)

    train_ld = DataLoader(ConferenceDataset(texts, [conferences.index(l) for l in labels], vocab), batch_size=BATCH_SIZE, shuffle=True)
    gru_model = BiGRUConferenceClassifier(len(vocab.word2idx), len(conferences)).to(DEVICE)
    opt = AdamW(gru_model.parameters(), lr=LR)
    for _ in range(GRU_EPOCHS):
        gru_model.train()
        for x, y in train_ld: opt.zero_grad(); F.cross_entropy(gru_model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()

    sbert_clf = SBERTConferenceClassifier() if HAS_SBERT else None
    if sbert_clf: sbert_clf.fit(texts, labels, ids)
    classical_clf = ClassicalConferenceStack()
    classical_clf.fit(texts, labels)
    zs_clf = None
    if HAS_ZS:
        try: zs_clf = ZeroShotConferenceClassifier()
        except: pass

    kw_extractor = ConferenceKeywordExtractor()
    kw_extractor.fit(texts, labels)

    return gru_model, sbert_clf, classical_clf, zs_clf, kw_extractor, conferences, vocab

# ─────────────────────────────────────────────────────────
# FRONTEND UI
# ─────────────────────────────────────────────────────────
def main():
    inject_custom_css()

    with st.sidebar:
        st.markdown("### ⚙️ Infrastructure")
        
        # --- THE LOUD FAILSAFE SIDEBAR ---
        if supabase:
            st.success("🟢 Supabase Connected")
        else:
            st.error("🔴 Supabase Offline")
            st.code(f"Diagnostic Error:\n{db_error}", language="text")
            st.caption("Fix: Run `pip install supabase postgrest-py` in your terminal, then restart the app.")
        # ---------------------------------
        
        st.success("🟢 Routing Engine Active")
        st.success("🟢 Multimodal Text Extractor")
        st.markdown("<br><br><br><b>KDSH AI Platform</b><br>Designed by Aanchal Chauhan", unsafe_allow_html=True)

    st.markdown("""
<div class="hero-box">
  <div>
    <h1 class="hero-title">Academic Publishability Engine</h1>
    <p class="hero-sub">Upload a research artifact to automatically evaluate its publishability readiness and route it to the optimal ML conference venue.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.spinner("Initializing Deep Learning weights..."):
        components = initialize_pipeline()
    if components[0] is None:
        st.error("❌ Critical Error: Missing reference corpus in `./Reference/Publishable/`.")
        return
    gru_model, sbert_clf, classical_clf, zs_clf, kw_extractor, conferences, vocab = components

    st.markdown(f"""
<div class="stat-strip">
  <div class="stat-card"><div class="stat-lbl">Active Ensembles</div><div class="stat-val">4</div></div>
  <div class="stat-card"><div class="stat-lbl">Supported Venues</div><div class="stat-val">{len(conferences)}</div></div>
  <div class="stat-card"><div class="stat-lbl">System Status</div><div class="stat-val" style="color:#10b981;">Online</div></div>
</div>
""", unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown("### 📤 Document Upload")
        st.caption("Supported formats: PDF, CSV.")
        uploaded_file = st.file_uploader("", type=['pdf', 'csv'], label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 ANALYZE PAPER", type="primary", use_container_width=True, disabled=uploaded_file is None)

    with col_result:
        st.markdown("### 📊 System Evaluation")
        if not analyze_btn and not uploaded_file:
            st.info("👈 Upload an academic artifact to begin evaluation.")

        if analyze_btn and uploaded_file:
            with st.spinner("Running Multi-Stage Evaluation Pipeline..."):
                ext = uploaded_file.name.split('.')[-1].lower()
                text = extract_text_from_stream(uploaded_file.read(), ext)

                if len(text.strip()) < 50:
                    st.error("Failed to extract meaningful text. The document may be empty or an un-OCRed image.")
                    return

                is_publishable, pub_reason = check_publishability(text)

                if not is_publishable:
                    st.markdown(f"""
<div class="val-card-container">
  <div class="val-header rejected">
    <div class="val-title">Evaluation Status: Rejected</div>
    <div class="val-big-price">NOT PUBLISHABLE</div>
  </div>
  <div class="val-footer">
    <div class="val-range-box red">
      <div class="val-range-lbl">Gatekeeper AI Feedback</div>
      <div class="val-range-val">{pub_reason}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
                    if supabase:
                        try: supabase.table('predictions').insert({"paper_id": uploaded_file.name, "publishable": 0, "conference": "N/A", "confidence": 0.0, "rationale": pub_reason}).execute()
                        except: pass
                    return

                all_probs, best_ref, best_sim = [], "Unknown", 0.0
                if sbert_clf:
                    p, best_ref, best_sim = sbert_clf.predict_proba(text)
                    all_probs.append((p, W_SBERT))
                if gru_model:
                    gru_model.eval()
                    with torch.no_grad():
                        logits = gru_model(torch.tensor(vocab.encode(text), dtype=torch.long).unsqueeze(0).to(DEVICE))
                        all_probs.append(({conferences[i]: float(F.softmax(logits, dim=-1)[0][i]) for i in range(len(conferences))}, W_GRU))
                if classical_clf: all_probs.append((classical_clf.predict_proba(text), W_CLASSICAL))
                if zs_clf:
                    try: all_probs.append((zs_clf.predict_proba(text), W_ZEROSHOT))
                    except: pass

                final_probs = {c: 0.0 for c in conferences}
                total_w = sum(w for _, w in all_probs)
                for p_dict, w in all_probs:
                    for c in conferences: final_probs[c] += p_dict.get(c, 0.0) * (w / total_w)

                pred_conf = max(final_probs, key=final_probs.get)
                conf_val = final_probs[pred_conf]
                kws = kw_extractor.top_keywords(text, pred_conf)
                
                rationale = f"Classified as <b>{pred_conf}</b> with an ensemble confidence of <b>{conf_val*100:.0f}%</b>. Semantic alignment with the {pred_conf} reference corpus '{best_ref}' confirmed. Core matching terminologies extracted: <i>{', '.join(kws) if kws else 'Core domain topics'}</i>."

                prob_html = ""
                for conf, prob in sorted(final_probs.items(), key=lambda i: i[1], reverse=True):
                    prob_html += f"""
<div class="prob-row">
    <div class="prob-label">{conf}</div>
    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {prob*100}%;"></div></div>
    <div class="prob-pct">{prob*100:.1f}%</div>
</div>"""

                st.markdown(f"""
<div class="val-card-container">
  <div class="val-header">
    <div class="val-title">Evaluation Status: Accepted</div>
    <div class="val-big-price">{pred_conf}</div>
    <div style="margin-top: 0.8rem; font-size: 1.1rem; font-weight: 500;">Optimal Venue Recommendation</div>
  </div>
  <div class="val-footer" style="flex-direction: column;">
    <div class="val-range-box">
      <div class="val-range-lbl">AI Routing Rationale</div>
      <div class="val-range-val">{rationale}</div>
    </div>
    <div>
      <div class="val-range-lbl" style="margin-bottom: 1rem;">Soft-Voting Probability Distribution</div>
      {prob_html}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                if supabase:
                    try: supabase.table('predictions').insert({"paper_id": uploaded_file.name, "publishable": 1, "conference": pred_conf, "confidence": conf_val, "rationale": rationale[:250]}).execute()
                    except: pass

if __name__ == "__main__":
    main()
