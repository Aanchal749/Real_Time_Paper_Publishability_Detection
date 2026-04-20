import fitz  # PyMuPDF
import json
from pathlib import Path

# Configuration
REFERENCE_DIR = "./Reference/Publishable"
OUTPUT_FILE = "references.jsonl"

def clean_text(text: str) -> str:
    """Removes excessive newlines and spaces to create a cleaner embedding."""
    text = text.replace('\n', ' ')
    return ' '.join(text.split())

def parse_references():
    ref_path = Path(REFERENCE_DIR)
    
    if not ref_path.exists():
        print(f"❌ Error: Could not find {REFERENCE_DIR}")
        return

    processed_count = 0
    
    # Open a JSON Lines file for writing
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        
        # Iterate through each conference folder (e.g., CVPR, NeurIPS)
        for conf_dir in ref_path.iterdir():
            if not conf_dir.is_dir():
                continue
                
            conference_name = conf_dir.name.lower()
            
            # Read each PDF in the conference folder
            for pdf_path in conf_dir.glob("*.pdf"):
                try:
                    doc = fitz.open(pdf_path)
                    text = "".join(page.get_text() for page in doc)
                    doc.close()
                    
                    if not text.strip():
                        print(f"⚠️ Warning: {pdf_path.name} is empty or unreadable.")
                        continue
                        
                    # Build the Schema dictionary
                    # We store the text, and crucial metadata for Task 2
                    document_data = {
                        "doc": clean_text(text),
                        "metadata": {
                            "paper_id": pdf_path.stem,
                            "conference": conference_name,
                            "type": "reference"
                        }
                    }
                    
                    # Write as a JSON string on a new line
                    f.write(json.dumps(document_data) + '\n')
                    processed_count += 1
                    print(f"✅ Processed {pdf_path.name} -> {conference_name.upper()}")
                    
                except Exception as e:
                    print(f"❌ Failed to process {pdf_path.name}: {e}")

    print(f"\n🎉 Done! {processed_count} reference papers saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parse_references()