import os
import glob
from typing import List
import json
import io

from ..entities.Paper import Paper
from ..entities.ScienceParse import ScienceParse


def _normalize_inline_text(value: str) -> str:
    return value.replace("\n", " ")

def read_science_parse(paperid, title, abstract, scienceparse_dir):
    """Read parsed PDF JSON and return a ScienceParse object."""
    scienceparse_path = os.path.join(scienceparse_dir, f"{paperid}.pdf.json")
    try:
        scienceparse_file = io.open(scienceparse_path, "r", encoding="utf8")
        scienceparse_str = scienceparse_file.read()
        scienceparse_data = json.loads(scienceparse_str)
    except Exception:
        # If no parsed PDF JSON exists, return an empty object so the pipeline does not crash.
        return ScienceParse(title, abstract, {}, {}, {}, {}, {}, {}, [], [])

    sections = {}
    reference_years = {}
    reference_titles = {}
    reference_venues = {}
    reference_mention_contexts = {}
    reference_num_mentions = {}

    metadata = scienceparse_data["metadata"]

    if metadata["sections"] is not None:
        for sectid in range(len(metadata["sections"])):
            heading = metadata["sections"][sectid]["heading"]
            text = metadata["sections"][sectid]["text"]
            sections[str(heading)] = text

    for refid in range(len(metadata["references"])):
        reference_titles[refid] = metadata["references"][refid]["title"]
        reference_years[refid] = metadata["references"][refid]["year"]
        reference_venues[refid] = metadata["references"][refid]["venue"]

    for menid in range(len(metadata["referenceMentions"])):
        refid = metadata["referenceMentions"][menid]["referenceID"]
        context = metadata["referenceMentions"][menid]["context"]
        old_context = reference_mention_contexts.get(refid, "")
        reference_mention_contexts[refid] = old_context + "\t" + context
        count = reference_num_mentions.get(refid, 0)
        reference_num_mentions[refid] = count + 1

    authors = metadata["authors"]
    emails = metadata["emails"]
    return ScienceParse(
        title,
        abstract,
        sections,
        reference_titles,
        reference_venues,
        reference_years,
        reference_mention_contexts,
        reference_num_mentions,
        authors,
        emails,
    )

def load_papers_from_dir(paper_json_dir: str, scienceparse_dir: str) -> List[Paper]:
    """
    Loads papers and their parsed science files from the given directories.
    
    Args:
        paper_json_dir (str): Directory containing paper JSON metadata/reviews.
        scienceparse_dir (str): Directory containing parsed PDF JSONs.
        
    Returns:
        List[Paper]: List of instantiated Paper objects with loaded science parses.
    """
    paper_json_filenames = sorted(glob.glob(os.path.join(paper_json_dir, '*.json')))
    papers = []
    
    for paper_json_filename in paper_json_filenames:
        paper = Paper.from_json(paper_json_filename)
        if not paper:
            continue

        # Keep text cleanup explicit in preprocess stage rather than hidden in data model.
        paper.ABSTRACT = _normalize_inline_text(paper.ABSTRACT)
            
        paper.SCIENCEPARSE = read_science_parse(
            paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir
        )
        papers.append(paper)
        
    return papers
