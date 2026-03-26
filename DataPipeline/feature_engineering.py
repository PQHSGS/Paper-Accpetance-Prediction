import os
import sys
import pickle as pkl
import random
import time
# Add parent path to support models package
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from parsing import load_papers_from_dir
from normalization import normalize_text
from feature_extraction import count_words, extract_hand_features
from reporting import read_features, save_features_to_file

def main(args):
    if len(args) < 9:
        print("Usage:", args[0], "<paper-json-dir> <scienceparse-dir> <out-dir> <submission-year> <feature_output_file> <tfidf_vector_file> <max_vocab_size> <encoder> <hand-feature>")
        return -1

    paper_json_dirs = args[1].split(',')
    scienceparse_dirs = args[2].split(',')
    out_dir = args[3]
    feature_output_file = args[4]
    vect_file = args[5]
    max_vocab_size = False if args[6] == 'False' else int(args[6])
    encoder_type = False if args[7] == 'False' else str(args[7])
    use_hand = False if args[8] == 'False' else True

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    is_train = not os.path.isfile(feature_output_file)
    idToFeature = dict()

    if is_train:
        print('Loading vector file from scratch..')
    else:
        print('Loading features and vector file from existing...')
        idToFeature = read_features(feature_output_file)
        # Vector loading logic would go here

    # 1. Parsing Module
    t_start = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] Reading reviews and science parse from {len(paper_json_dirs)} directories...')
    papers = []
    
    for idx, (p_dir, s_dir) in enumerate(zip(paper_json_dirs, scienceparse_dirs)):
        t_ds = time.time()
        dataset_papers = load_papers_from_dir(p_dir, s_dir)
        dataset_root = os.path.dirname(os.path.dirname(p_dir))
        accepted_txt_path = os.path.join(dataset_root, 'acl_accepted.txt')
        accepted_titles = set()
        
        if os.path.exists(accepted_txt_path):
            with open(accepted_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    accepted_titles.add(line.strip().lower())
                    
            for p in dataset_papers:
                p.ACCEPTED = (p.get_title().strip().lower() in accepted_titles)
                
        papers.extend(dataset_papers)
        print(f'  [{time.strftime("%H:%M:%S")}] Dataset {idx+1}/{len(paper_json_dirs)}: {p_dir} -> {len(dataset_papers)} papers ({time.time()-t_ds:.1f}s)')
        
    random.shuffle(papers)
    print(f'[{time.strftime("%H:%M:%S")}] Total papers loaded: {len(papers)} ({time.time()-t_start:.1f}s)')

    def get_feature_id(feature_name: str) -> int:
        return idToFeature.get(feature_name)

    def add_feature_to_dict(fname: str) -> int:
        if fname not in idToFeature:
            fid = len(idToFeature)
            idToFeature[fname] = fid
        return idToFeature[fname]

    # 2. Normalization Module
    t_norm = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] Normalizing {len(papers)} papers...')
    paper_content_corpus = []
    for i, paper in enumerate(papers):
        content = paper.SCIENCEPARSE.get_paper_content()
        paper_content_corpus.append(normalize_text(content, only_char=True, lower=True, stop_remove=True))
        if (i+1) % 500 == 0 or (i+1) == len(papers):
            print(f'  [{time.strftime("%H:%M:%S")}] Normalized {i+1}/{len(papers)} papers ({time.time()-t_norm:.1f}s)')
    
    outCorpusFilename = os.path.join(out_dir, 'corpus.pkl')
    if not os.path.isfile(outCorpusFilename):
        paper_content_corpus_words = []
        for p_content in paper_content_corpus:
            paper_content_corpus_words += p_content.split(' ')
        pkl.dump(paper_content_corpus_words, open(outCorpusFilename, 'wb'))
    else:
        paper_content_corpus_words = pkl.load(open(outCorpusFilename, 'rb'))
    print(f'[{time.strftime("%H:%M:%S")}] Total words in normalized corpus: {len(paper_content_corpus_words)} ({time.time()-t_norm:.1f}s)')

    # 3. Feature extraction Module
    t_feat = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] Counting words and extracting features...')
    hfws, most_frequent_words, least_frequent_words = count_words(paper_content_corpus_words, 0.01, 0.05, 3)

    outLabelsFile = open(os.path.join(out_dir, f'labels_{max_vocab_size}_{encoder_type}_{use_hand}.tsv'), 'w')
    outIDFile = open(os.path.join(out_dir, f'ids_{max_vocab_size}_{encoder_type}_{use_hand}.tsv'), 'w')
    outSvmLiteFile = open(os.path.join(out_dir, f'features.svmlite_{max_vocab_size}_{encoder_type}_{use_hand}.txt'), 'w')

    print(f'[{time.strftime("%H:%M:%S")}] Writing SVMLite features for {len(papers)} papers...')
    for p_idx, paper in enumerate(papers, start=1):
        outIDFile.write(f"{p_idx}\t{paper.get_title()}\n")
        label = int(paper.get_accepted() == True)
            
        outLabelsFile.write(str(label) + "\n")

        # SVMLite Export
        combined_features = {}
        if use_hand:
            hand_features = extract_hand_features(paper, paper.SCIENCEPARSE, hfws, most_frequent_words, least_frequent_words)
            for feature_name, value in hand_features.items():
                if is_train:
                    fid = add_feature_to_dict(feature_name)
                else:
                    fid = get_feature_id(feature_name)
                
                if fid is not None and value != 0:
                    combined_features[fid] = value

        # Writing out row for SVMLite
        outSvmLiteFile.write(str(label) + " ")
        for fid in sorted(combined_features.keys()):
            outSvmLiteFile.write(f"{fid}:{combined_features[fid]} ")
        outSvmLiteFile.write("\n")
        
        if p_idx % 500 == 0 or p_idx == len(papers):
            print(f'  [{time.strftime("%H:%M:%S")}] Exported {p_idx}/{len(papers)} papers ({time.time()-t_feat:.1f}s)')

    if is_train:
        save_features_to_file(idToFeature, feature_output_file)

    outLabelsFile.close()
    outIDFile.close()
    outSvmLiteFile.close()
    total_time = time.time() - t_start
    print(f'[{time.strftime("%H:%M:%S")}] Done! Saved feature pipelines properly. Total time: {total_time:.1f}s')

if __name__ == "__main__":
    main(sys.argv)
