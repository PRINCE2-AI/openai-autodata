import argparse
import urllib.request
from pathlib import Path
from typing import List
from urllib.error import URLError

import arxiv


def download_cs_papers(max_results: int = 10, output_dir: str = "data/papers") -> List[str]:
    """Download recent AI, ML, and NLP papers from arXiv."""
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    downloaded = []
    for paper in arxiv.Client().results(search):
        safe_id = paper.get_short_id().replace("/", "_")
        filepath = destination / f"{safe_id}.pdf"
        if filepath.exists():
            print(f"Already exists: {filepath.name}")
            continue

        print(f"Downloading: {paper.title[:70]}...")
        try:
            urllib.request.urlretrieve(paper.pdf_url, filepath)
        except (OSError, URLError) as exc:
            print(f"Download failed for {paper.get_short_id()}: {exc}")
            continue
        print(f"Saved: {filepath}")
        downloaded.append(str(filepath))

    print(f"Downloaded {len(downloaded)} new paper(s).")
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download recent computer-science papers from arXiv.")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--output-dir", default="data/papers")
    arguments = parser.parse_args()
    download_cs_papers(max_results=arguments.max_results, output_dir=arguments.output_dir)
