"""Download ISMIR papers (PDF + metadata) and extract abstract, discussion and conclusion."""

import argparse
import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

URL = "https://ismir.net/conferences/ismir-{year}/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research scraper; UPF meta-research project)"}
FIELDS = ["year", "paper_id", "title", "authors", "pages", "doi", "pdf_url", "pdf_path",
          "abstract", "discussion", "conclusion"]

# ISMIR headings are uppercase lines like "5. DISCUSSION AND CONCLUSION"; case-sensitive so body text never matches.
SECTIONS = {
    "abstract": "ABSTRACT",
    "discussion": "[A-Z ]*DISCUSSION[A-Z ]*",
    "conclusion": "[A-Z ]*CONCLUSIONS?[A-Z ]*",
}
NEXT_HEADING = re.compile(r"\n\s*(?:\d+\.?\s+[A-Z][A-Z ,&\-]{3,}|REFERENCES|ACKNOWLEDGE?MENTS?|ETHICS STATEMENT)\s*\n")
# Page footer, e.g. "Proceedings of the 26th ISMIR Conference, Daejeon, Korea, September 21-25, 2025 16"
FOOTER = re.compile(r"^Proceedings of the \d+\w* (?:ISMIR|International Society).*$", re.MULTILINE)


def get_papers(year):
    """Each paper on the conference page is a title/authors block followed by a 'Download pdf' button."""
    soup = BeautifulSoup(requests.get(URL.format(year=year), headers=HEADERS, timeout=30).text, "html.parser")
    papers = {}
    for button in soup.select("a.et_pb_button[href$='.pdf']"):
        url = button["href"]
        if "Proceedings" in url or url in papers:  # skip the full proceedings and duplicate links
            continue
        info = button.find_previous("div", class_="et_pb_text_inner")
        link = info.find("a")
        title, pages = re.match(r"(.*?)\s*(\d+\s*-\s*\d+)?$", link.get_text(" ", strip=True)).groups()
        papers[url] = {
            "year": year,
            "paper_id": f"{year}_{Path(url).stem}",
            "title": title,
            "authors": " ".join(info.find_all("span")[-1].get_text(" ").split()),
            "pages": (pages or "").replace(" ", ""),
            "doi": link["href"] if "doi.org" in link["href"] else "",
            "pdf_url": url,
        }
    return list(papers.values())


def extract_section(text, heading):
    match = re.search(rf"(?:^|\n)\s*(?:\d+\.?\s+)?{heading}\s*\n", text)
    if not match:
        return ""
    rest = text[match.end():]
    end = NEXT_HEADING.search(rest)
    section = " ".join(rest[: end.start() if end else 5000].split())
    return re.sub(r"\s+\d+$", "", section)  # drop trailing page number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=range(2020, 2026))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, help="max papers per year (for testing)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        papers = get_papers(year)[: args.limit]
        print(f"ISMIR {year}: {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            pdf = args.out / "pdfs" / str(year) / f"{paper['paper_id']}.pdf"
            try:
                if not pdf.exists() or pdf.stat().st_size == 0:
                    resp = requests.get(paper["pdf_url"], headers=HEADERS, timeout=60)
                    resp.raise_for_status()
                    pdf.parent.mkdir(parents=True, exist_ok=True)
                    pdf.write_bytes(resp.content)
                    time.sleep(1)  # be polite to the servers
                text = FOOTER.sub("", "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages))
                paper.update({name: extract_section(text, h) for name, h in SECTIONS.items()}, pdf_path=str(pdf))
            except Exception as e:
                print(f"  ! {paper['paper_id']}: {e}")
            print(f"  [{i}/{len(papers)}] {paper['title'][:70]}")

        # One CSV per year: re-running a year replaces only that year's file.
        csv_path = args.out / f"metadata_{year}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(papers)
        print(f"Saved {len(papers)} papers to {csv_path}")


if __name__ == "__main__":
    main()
