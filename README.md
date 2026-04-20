<<<<<<< HEAD
# Real_Time_Paper_Publishability_Detection
The Real-Time Paper Publishability Detection System is an AI-powered platform that evaluates whether a research paper is suitable for publication in real time. It leverages Natural Language Processing, machine learning, and academic quality metrics to analyze research manuscripts and provide instant feedback on their publishability.
=======
# KDSH 2025 — Pathway Hackathon Solution

## Your Dataset Structure
```
drive-download.../
├── Papers/
│   ├── P001.pdf ... P150.pdf
└── Reference/
    ├── Non-Publishable/  *.pdf
    └── Publishable/
        ├── CVPR/
        ├── EMNLP/
        ├── KDD/
        ├── NeurIPS/
        └── TMLR/
```

## Run

### Step 1 — Publishability
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python task1_publishability.py --papers_dir ./Papers --reference_dir ./Reference --output results_task1.csv
```

### Step 2 — Conference Selection (local, no GDrive needed)
```bash
python task2_conference.py --task1_csv results_task1.csv --papers_dir ./Papers --reference_dir ./Reference --output results_task2.csv
```

### Step 2 — Conference Selection (Pathway + Google Drive)
```bash
export OPENAI_API_KEY="sk-..."
python task2_conference.py --task1_csv results_task1.csv --papers_dir ./Papers --gdrive_folder_id FOLDER_ID --service_account service_account.json --output results_task2.csv
```

### Step 3 — Final CSV
```bash
python merge_results.py --task2_csv results_task2.csv --output results.csv
```

## Install
```bash
pip install anthropic pymupdf pathway requests scikit-learn openai
```
Note: Pathway requires Linux/macOS. Windows users: use WSL or Docker.
>>>>>>> 6fa1e40 (Initial commit)
