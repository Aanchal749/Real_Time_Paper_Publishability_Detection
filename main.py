"""
KDSH 2025 - Unified Evaluation Pipeline
Integrates Task 1 (Publishability) and Task 2 (Conference Selection via Pathway RAG).
"""

import json
import csv
import time
import sys
import os
import getpass
from pathlib import Path

import anthropic
import fitz  # PyMuPDF
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
from pathway.xpacks.llm.embedders import OpenAIEmbedder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
PAPERS_DIR    = "./Papers"
OUTPUT_CSV    = "results.csv" # Required hackathon filename

CLAUDE_MODEL  = "claude-3-haiku-20240307" # Standardizing model across tasks
MAX_TOKENS    = 1800
PDF_MAX_CHARS = 14000
MAX_RETRIES   = 3

# Pathway GDrive Config
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "YOUR_GDRIVE_FOLDER_ID")
GDRIVE_CREDS     = os.environ.get("GDRIVE_CREDS", "credentials.json")

print("=" * 60)
print("  KDSH 2025 — Unified Application Pipeline")
print("=" * 60)

# API Keys Setup
api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not api_key:
    api_key = getpass.getpass("Enter Anthropic API key: ").strip()

openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not openai_key:
    openai_key = getpass.getpass("Enter OpenAI API key (for embeddings): ").strip()
    os.environ["OPENAI_API_KEY"] = openai_key

try:
    client = anthropic.Anthropic(api_key=api_key)
except Exception as e:
    raise SystemExit(f"❌ Connection failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PATHWAY VECTORSTORE SETUP (Streaming from Google Drive)
# ─────────────────────────────────────────────────────────────────────────────
print("\nInitializing Pathway VectorStore via Google Drive...")

class DocumentInputSchema(pw.Schema):
    doc: str
    metadata: dict

# Fallback in case the Google Drive credentials aren't set up yet
if os.path.exists(GDRIVE_CREDS):
    print("✅ Found GDrive credentials. Streaming from Google Drive...")
    reference_data = pw.io.gdrive.read(
        object_id=GDRIVE_FOLDER_ID,
        service_user_credentials_file=GDRIVE_CREDS,
        with_metadata=True,
        mode="streaming"
    )
else:
    print("⚠️ GDrive credentials not found. Falling back to local './Reference/Publishable/' folder.")
    reference_data = pw.io.fs.read(
        "./Reference/Publishable/",
        format="json", 
        schema=DocumentInputSchema
    )

# Setup Embedder for the VectorStore
embedder = OpenAIEmbedder()

# Create and start the Pathway server on a background thread
server = VectorStoreServer(
    *reference_data,
    embedder=embedder,
)

# Run server in background on port 8000
print("Starting VectorStore server...")
server.run_server(host="127.0.0.1", port=8000, threaded=True)

# Increased sleep time to ensure the local server finishes indexing before we query it
print("Indexing documents... please wait.")
time.sleep(5) 

# Initialize the Client used to query the server inside the loop
rag_client = VectorStoreClient(host="127.0.0.1", port=8000)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def extract_pdf(path: str) -> str:
    try:
        doc  = fitz.open(path)
        text = "".join(p.get_text() for p in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  [WARN] Cannot read {path}: {e}")
        return ""

def build_paper_summary(text: str, max_chars: int = PDF_MAX_CHARS) -> str:
    if len(text) <= max_chars: return text
    front = text[:5000]
    back  = text[-3000:]
    middle = text[(len(text) // 2 - 1500):(len(text) // 2 + 1500)]
    return (front + "\n\n[...]\n\n" + middle + "\n\n[...]\n\n" + back)[:max_chars]

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def assess_publishability(paper_text: str, paper_id: str) -> int:
    """Task 1 logic adapted for unified pipeline."""
    prompt = f"""
    Evaluate the following research paper for publishability. 
    1 = Publishable (sound methodology, clear writing, genuine contribution).
    0 = Non-Publishable (flawed methodology, incoherent, inflated claims).
    
    Paper Text:
    {build_paper_summary(paper_text)}
    
    Respond strictly with JSON: {{"publishable": 1}} or {{"publishable": 0}}
    """
    
    for _ in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text.strip()
            
            # SAFE JSON EXTRACTION: No regex used to prevent copy/paste cut-offs
            raw = raw.replace("```json", "").replace("```", "").strip()
            
            return int(json.loads(raw).get("publishable", 0))
        except Exception:
            time.sleep(2)
    return 0

def get_conference_and_rationale(target_paper_text: str) -> tuple:
    """Task 2 logic querying the Pathway index via Client."""
    try:
        # Query the local Pathway Server using the Client
        nearest_neighbors = rag_client(query=target_paper_text, k=3)
        
        if not nearest_neighbors:
            return "na", "Could not retrieve neighbors."
            
        conferences = [neighbor['metadata'].get('conf', 'na') for neighbor in nearest_neighbors]
        predicted_conf = max(set(conferences), key=conferences.count)
        best_reference = nearest_neighbors[0]['text']
        
        prompt = f"""
        A new paper has been matched to '{predicted_conf.upper()}'.
        Target Paper: {target_paper_text[:2000]}
        Reference Paper in {predicted_conf.upper()}: {best_reference[:1000]}
        
        Write a formal rationale (UNDER 100 WORDS) explaining how the target paper aligns 
        with the themes of {predicted_conf.upper()}.
        """
        
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return predicted_conf, response.content[0].text.strip()
    except Exception as e:
        return "na", f"Error during RAG pipeline: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    papers_path = Path(PAPERS_DIR)
    papers_path.mkdir(exist_ok=True) # Prevent crash if directory is missing
    pdfs = sorted(papers_path.glob("*.pdf"))
    
    if not pdfs:
        print(f"⚠️ No PDFs found in {PAPERS_DIR}. Please add some PDFs to test.")
        sys.exit()

    results = []
    
    print(f"\nProcessing {len(pdfs)} papers...\n")
    
    for i, pdf_path in enumerate(pdfs):
        paper_id = pdf_path.stem
        print(f"[{i+1}/{len(pdfs)}] {paper_id} ... ", end="", flush=True)
        
        text = extract_pdf(str(pdf_path))
        if not text:
            print("FAILED TO READ PDF")
            results.append({"Paper ID": paper_id, "Publishable": 0, "Conference": "na", "Rationale": "na"})
            continue
            
        # Task 1: Publishability
        is_publishable = assess_publishability(text, paper_id)
        
        # Task 2: Conditional Conference Routing
        if is_publishable == 1:
            print("✅ PUBLISHABLE -> Finding Conference... ", end="")
            conf, rationale = get_conference_and_rationale(text)
            print(f"[{conf.upper()}]")
        else:
            print("❌ NON-PUBLISHABLE")
            conf, rationale = "na", "na"
            
        results.append({
            "Paper ID": paper_id,
            "Publishable": is_publishable,
            "Conference": conf,
            "Rationale": rationale
        })

    # Save to the specific format requested by the Hackathon
    fieldnames = ["Paper ID", "Publishable", "Conference", "Rationale"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE. Results saved to {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()