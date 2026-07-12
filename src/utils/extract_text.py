import argparse
from pathlib import Path

import fitz


def extract_text_from_pdf(pdf_path: str, max_chars: int = 8000) -> str:
    """Extract up to max_chars of text from a PDF."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")

    try:
        with fitz.open(pdf_path) as document:
            chunks = []
            length = 0
            for page in document:
                page_text = page.get_text()
                chunks.append(page_text)
                length += len(page_text)
                if length >= max_chars:
                    break
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not read {pdf_path}: {exc}")
        return ""
    return "".join(chunks)[:max_chars]


def process_all_papers(
    papers_dir: str = "data/papers",
    output_dir: str = "data/raw_text",
    max_chars: int = 8000,
) -> int:
    """Extract text from every PDF in papers_dir and return the success count."""
    source = Path(papers_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(source.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s).")

    extracted_count = 0
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(str(pdf_path), max_chars=max_chars)
        if not text:
            print(f"Skipped: {pdf_path.name} (no extractable text)")
            continue

        text_path = destination / f"{pdf_path.stem}.txt"
        text_path.write_text(text, encoding="utf-8")
        extracted_count += 1
        print(f"Extracted: {pdf_path.name} -> {len(text)} characters")

    print(f"Extracted {extracted_count} paper(s).")
    return extracted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from downloaded PDF papers.")
    parser.add_argument("--papers-dir", default="data/papers")
    parser.add_argument("--output-dir", default="data/raw_text")
    parser.add_argument("--max-chars", type=int, default=8000)
    arguments = parser.parse_args()
    process_all_papers(
        papers_dir=arguments.papers_dir,
        output_dir=arguments.output_dir,
        max_chars=arguments.max_chars,
    )
