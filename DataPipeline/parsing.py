import os
import glob
from typing import List

from models.Paper import Paper
from models.ScienceParseReader import ScienceParseReader

def load_papers_from_dir(paper_json_dir: str, scienceparse_dir: str) -> List[Paper]:
    """
    Loads papers and their parsed science files from the given directories.
    
    Args:
        paper_json_dir (str): Directory containing paper JSON metadata/reviews.
        scienceparse_dir (str): Directory containing parsed PDF JSONs.
        
    Returns:
        List[Paper]: List of instantiated Paper objects with loaded science parses.
    """
    paper_content_corpus = []
    paper_json_filenames = sorted(glob.glob(os.path.join(paper_json_dir, '*.json')))
    papers = []
    
    for paper_json_filename in paper_json_filenames:
        paper = Paper.from_json(paper_json_filename)
        if not paper:
            continue
            
        paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(
            paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir
        )
        papers.append(paper)
        
    return papers
