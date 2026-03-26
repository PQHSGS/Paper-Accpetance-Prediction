import os
import sys

# Ensure DataPipeline is in path
sys.path.insert(1, os.path.join(os.getcwd(), 'DataPipeline'))

from feature_engineering import main as run_extraction

datasets = [
    "acl_2017",
    "iclr_2017",
    "arxiv.cs.ai_2007-2017",
    "arxiv.cs.cl_2007-2017",
    "arxiv.cs.lg_2007-2017",
    "conll_2016"
]

all_vocab = "False"
encoder = "w2v"
hand = "True"
submission_year = "2017" # Dummy value not really used strongly in scripts

# 1. Ensure target dirs exist
for split in ["train", "dev", "test"]:
    os.makedirs(f"Dataset/all_combined/{split}/dataset", exist_ok=True)

# 2. Iterate through splits and extract features
for split in ["dev", "test", "train"]:
    print(f"\n--- Extracting Features for {split.upper()} split ---")
    paper_json_dirs = []
    scienceparse_dirs = []
    
    for ds in datasets:
        rj_dir = f"Dataset/{ds}/{split}/reviews"
        sp_dir = f"Dataset/{ds}/{split}/parsed_pdfs"
        if os.path.isdir(rj_dir) and os.path.isdir(sp_dir):
            paper_json_dirs.append(rj_dir)
            scienceparse_dirs.append(sp_dir)
            
    out_dir = f"Dataset/all_combined/{split}/dataset"
    feature_vocab_file = f"Dataset/all_combined/train/dataset/features_{all_vocab}_{encoder}_{hand}.dat"
    vector_vocab_file = f"Dataset/all_combined/train/dataset/vectors_{all_vocab}_{encoder}.txt"
    
    args = [
        "feature_engineering.py",
        ",".join(paper_json_dirs),   # arg 1
        ",".join(scienceparse_dirs), # arg 2
        out_dir,                     # arg 3
        feature_vocab_file,          # arg 4
        vector_vocab_file,           # arg 5
        all_vocab,                   # arg 6
        encoder,                     # arg 7
        hand                         # arg 8
    ]
    
    run_extraction(args)
