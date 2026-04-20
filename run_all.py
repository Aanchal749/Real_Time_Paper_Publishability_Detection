"""
KDSH 2025 - All-in-One Pipeline (Google Gemini Version)
=========================================================
Just run:  python run_all.py
It will ask for your Gemini API key, then do everything:
  1. Read all PDFs from Papers/
  2. Load reference papers from Reference/
  3. Classify each paper (Publishable / Non-Publishable)
  4. Select conference for publishable papers
  5. Save results.csv
"""

import json
import csv
import time
import re
import sys
import os
from pathlib import Path
from collections import Counter

# ── Check and install dependencies ───────────────────────────────────────────
print("=" * 60)
print("  KDSH 2025 - Research Paper Evaluation Pipeline")
print("  Powered by Google Gemini")
print("=" * 60)
print()

try:
    import fitz
except ImportError:
    print("Installing PyMuPDF...")
    os.system(f"{sys.executable} -m pip install PyMuPDF")
    import fitz

try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    os.system(f"{sys.executable} -m pip install google-generativeai")
    import google.generativeai as genai

print("✅ All dependencies ready.\n")

# ── Ask for API key ───────────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 1: Enter your Google Gemini API Key")
print("  Get it from: https://aistudio.google.com → Get API Key")
print("─" * 60)

while True:
    api_key = input("\nPaste your Gemini API key here: ").strip()

    if not api_key:
        print("❌ No key entered. Please try again.")
        continue

    # Test the key
    print("\nTesting API key...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        test  = model.generate_content("Say hi in one word")
        print(f"✅ API key works! Response: {test.text.strip()}\n")
        break
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            print("⚠️  Rate limit hit — waiting 60 seconds...")
            time.sleep(60)
            print("Retrying...")
            continue
        else:
            print(f"❌ Error: {e}")
            print("   Check your internet connection and try again.")

# ── Ask for folder paths ──────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 2: Confirm folder paths")
print("─" * 60)

def ask_path(prompt, default):
    val = input(f"\n{prompt}\n(press Enter to use '{default}'): ").strip()
    return val if val else default

PAPERS_DIR    = ask_path("Papers folder path?", "./Papers")
REFERENCE_DIR = ask_path("Reference folder path?", "./Reference")
OUTPUT_CSV    = "results.csv"

# Validate
if not Path(PAPERS_DIR).exists():
    raise SystemExit(f"\n❌ Papers folder not found: {PAPERS_DIR}\n")
if not Path(REFERENCE_DIR).exists():
    raise SystemExit(f"\n❌ Reference folder not found: {REFERENCE_DIR}\n")

pdfs = sorted(Path(PAPERS_DIR).glob("*.pdf"))
if not pdfs:
    raise SystemExit(f"\n❌ No PDF files found in {PAPERS_DIR}\n")

print(f"\n✅ Found {len(pdfs)} papers in {PAPERS_DIR}")
print(f"✅ Reference folder: {REFERENCE_DIR}")
print(f"✅ Output will be saved to: {OUTPUT_CSV}\n")

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL      = "gemini-2.0-flash"   # free tier model
PDF_MAX_CHARS     = 10000
RETRY_DELAY       = 60   # wait 60s when quota hit
MAX_RETRIES       = 3
VALID_CONFERENCES = ["cvpr", "neurips", "emnlp", "kdd", "tmlr", "daa"]

CONFERENCE_INFO = {
    "cvpr"   : "Computer Vision and Pattern Recognition — image/video, visual learning, 3D vision",
    "neurips": "Neural Information Processing Systems — ML theory, deep learning, AI, optimization",
    "emnlp"  : "Empirical Methods in NLP — natural language processing, language models, linguistics",
    "kdd"    : "Knowledge Discovery & Data Mining — large-scale data mining, graphs, applied ML",
    "tmlr"   : "Transactions on ML Research — broad ML, reproducibility, rigorous methodology",
    "daa"    : "Design & Analysis of Algorithms — algorithms, complexity, combinatorics, theory",
}

gemini = genai.GenerativeModel(GEMINI_MODEL)

# ── PDF helpers ───────────────────────────────────────────────────────────────
def extract_pdf(path):
    try:
        doc  = fitz.open(path)
        text = "".join(p.get_text() for p in doc)
        doc.close()
        return text.strip()
    except:
        return ""

def truncate(text, max_chars=PDF_MAX_CHARS):
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]

# ── Load reference papers ─────────────────────────────────────────────────────
print("─" * 60)
print("  STEP 3: Loading reference papers...")
print("─" * 60)

pub_refs     = []
non_pub_refs = []

non_pub_path = Path(REFERENCE_DIR) / "Non-Publishable"
if non_pub_path.exists():
    for pdf in sorted(non_pub_path.glob("*.pdf")):
        text = extract_pdf(str(pdf))
        if text:
            non_pub_refs.append({"name": pdf.name, "text": truncate(text, 1500), "conf": "na"})
    print(f"  Non-Publishable refs : {len(non_pub_refs)}")
else:
    print(f"  [WARN] Not found: {non_pub_path}")

pub_path = Path(REFERENCE_DIR) / "Publishable"
if pub_path.exists():
    for conf_dir in sorted(pub_path.iterdir()):
        if not conf_dir.is_dir():
            continue
        count = 0
        for pdf in sorted(conf_dir.glob("*.pdf")):
            text = extract_pdf(str(pdf))
            if text:
                pub_refs.append({
                    "name": pdf.name,
                    "text": truncate(text, 1500),
                    "conf": conf_dir.name.lower()
                })
                count += 1
        print(f"  {conf_dir.name:12s} refs : {count}")
else:
    print(f"  [WARN] Not found: {pub_path}")

print(f"\n  Total: {len(pub_refs)} publishable + {len(non_pub_refs)} non-publishable refs\n")

# Build few-shot examples
few_shot_parts = []
for r in non_pub_refs[:2]:
    few_shot_parts.append(
        f"--- EXAMPLE ---\nPaper: {r['name']}\n"
        f"Excerpt:\n{r['text'][:800]}\nDecision: Non-Publishable\n"
    )
for r in pub_refs[:2]:
    few_shot_parts.append(
        f"--- EXAMPLE ---\nPaper: {r['name']} [Conference: {r['conf'].upper()}]\n"
        f"Excerpt:\n{r['text'][:800]}\nDecision: Publishable\n"
    )
few_shot = "\n".join(few_shot_parts)

# ── Gemini API call ───────────────────────────────────────────────────────────
def call_gemini(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = gemini.generate_content(prompt)
            raw  = resp.text.strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)

        except json.JSONDecodeError:
            # Try to extract JSON from response
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except:
                    pass
            print(f"\n  [WARN] Could not parse JSON (attempt {attempt+1})")

        except Exception as e:
            err = str(e)
            if "quota" in err.lower() or "429" in err:
                print(f"\n  [Rate limit] Waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            elif "500" in err or "503" in err:
                print(f"\n  [Server error] Waiting 5s...")
                time.sleep(5)
            else:
                print(f"\n  [Error attempt {attempt+1}]: {e}")
                time.sleep(3)
    return None


# ── Retrieve similar reference papers ────────────────────────────────────────
def retrieve_similar(query, k=5):
    stopwords = {"the","a","an","and","or","of","in","to","for",
                 "is","are","was","with","that","this","we","our"}
    q_words = set(re.findall(r'\b\w+\b', query.lower())) - stopwords
    scored  = []
    for doc in pub_refs:
        d_words = set(re.findall(r'\b\w+\b', doc["text"].lower()))
        scored.append((len(q_words & d_words), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


# ── Prompts ───────────────────────────────────────────────────────────────────
PUBLISH_PROMPT_TEMPLATE = """You are an expert academic peer reviewer.

Label NON-PUBLISHABLE (0) if paper has any of:
- Inappropriate or unjustified methodology
- Incoherent or illogical arguments  
- Unrealistic or unsubstantiated claims / inflated results
- No meaningful novelty or contribution
- Flawed experiments or missing baselines

Label PUBLISHABLE (1) if paper:
- Has a clear, well-motivated problem
- Uses sound, justified methodology
- Has realistic, reproducible results
- Makes a genuine contribution

Reference examples:
{few_shot}

--- PAPER TO EVALUATE ---
{paper_text}

Reply ONLY with valid JSON, no extra text:
{{
  "publishable": 0 or 1,
  "confidence": "low" or "medium" or "high",
  "key_issues": ["..."],
  "strengths": ["..."],
  "rationale": "2-3 sentences"
}}"""

CONFERENCE_PROMPT_TEMPLATE = """You are an expert academic program committee member.

Available conferences:
{conf_list}

Benchmark reference papers for context:
{context}

--- SUBMITTED PAPER ---
{paper_text}

Choose the SINGLE best conference. Write rationale in STRICTLY ≤100 words.

Reply ONLY with valid JSON, no extra text:
{{
  "conference": "cvpr or neurips or emnlp or kdd or tmlr or daa",
  "confidence": "low or medium or high",
  "rationale": "≤100 word explanation"
}}"""


# ── Main pipeline ─────────────────────────────────────────────────────────────
print("─" * 60)
print(f"  STEP 4: Processing {len(pdfs)} papers...")
print("─" * 60 + "\n")

results   = []
pub_count = 0

for i, pdf_path in enumerate(pdfs):
    paper_id = pdf_path.stem
    print(f"[{i+1:3d}/{len(pdfs)}] {paper_id} ", end="", flush=True)

    # Extract text from PDF
    text = extract_pdf(str(pdf_path))
    if not text:
        print("→ SKIP (cannot read PDF)")
        results.append({"Paper ID": paper_id, "Publishable": 0,
                        "Conference": "na", "Rationale": "na"})
        continue

    # ── Task 1: Publishability ────────────────────────────────────────────────
    pub_prompt = PUBLISH_PROMPT_TEMPLATE.format(
        few_shot   = few_shot,
        paper_text = truncate(text)
    )
    pub_result = call_gemini(pub_prompt)

    if pub_result is None:
        print("→ ERROR (API failed)")
        results.append({"Paper ID": paper_id, "Publishable": 0,
                        "Conference": "na", "Rationale": "na"})
        continue

    label = pub_result.get("publishable", 0)

    if label == 0:
        print(f"→ ❌ Non-Publishable [{pub_result.get('confidence','?')}]")
        results.append({"Paper ID": paper_id, "Publishable": 0,
                        "Conference": "na", "Rationale": "na"})

    else:
        pub_count += 1
        print(f"→ ✅ Publishable ", end="", flush=True)

        # ── Task 2: Conference selection ──────────────────────────────────────
        retrieved  = retrieve_similar(text[:600])
        ctx_parts  = [f"[{d['conf'].upper()}]\n{d['text'][:500]}" for d in retrieved]
        context    = "\n\n".join(ctx_parts) if ctx_parts else "No references found."
        conf_list  = "\n".join(f"- {k.upper()}: {v}" for k, v in CONFERENCE_INFO.items())

        conf_prompt = CONFERENCE_PROMPT_TEMPLATE.format(
            conf_list  = conf_list,
            context    = context,
            paper_text = truncate(text)
        )
        conf_result = call_gemini(conf_prompt)

        if conf_result is None:
            conf = "neurips"; rationale = "Could not determine."; confidence = "low"
        else:
            conf       = conf_result.get("conference", "neurips").lower()
            rationale  = conf_result.get("rationale", "")
            confidence = conf_result.get("confidence", "?")
            if conf not in VALID_CONFERENCES:
                conf = "neurips"
            # Enforce ≤100 words
            words = rationale.split()
            if len(words) > 100:
                rationale = " ".join(words[:100]) + "..."

        print(f"→ {conf.upper()} [{confidence}]")
        results.append({
            "Paper ID"   : paper_id,
            "Publishable": 1,
            "Conference" : conf,
            "Rationale"  : rationale
        })

    time.sleep(5)  # free tier = 15 requests/min, so wait 5s between papers

# ── Save results.csv ──────────────────────────────────────────────────────────
results.sort(key=lambda x: x["Paper ID"])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Paper ID","Publishable","Conference","Rationale"])
    writer.writeheader()
    writer.writerows(results)

# ── Final summary ─────────────────────────────────────────────────────────────
conf_counts = Counter(r["Conference"] for r in results if r["Publishable"] == 1)

print()
print("=" * 60)
print(f"  ✅  DONE!  Results saved to: {OUTPUT_CSV}")
print("=" * 60)
print(f"  Total papers       : {len(results)}")
print(f"  ✅ Publishable      : {pub_count}")
print(f"  ❌ Non-Publishable  : {len(results) - pub_count}")
if conf_counts:
    print("\n  Conference breakdown:")
    for c, n in sorted(conf_counts.items()):
        print(f"    {c.upper():10s} : {n} papers")
print("=" * 60)
print(f"\n  Your submission file is ready → {OUTPUT_CSV}\n")