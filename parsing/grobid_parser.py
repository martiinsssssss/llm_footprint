"""GROBID-based parser: PDF to cached TEI (GrobidParser) and TEI to Paper (TeiReader)."""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Iterator

import requests
from lxml import etree

from parsing.models import Paper, Section

GROBID_URL = "http://localhost:8070"
TEI_CACHE_DIR = Path("data/interim/tei")
REQUEST_TIMEOUT_SECONDS = 300

# PDF file names are `{paper_id}.pdf`, with paper_id like "2024_000001" (same as the CSVs).
PAPER_ID_PATTERN = re.compile(r"^(?P<year>\d{4})_\d{6}$")

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}

# Structural noise removed before extracting text: citation markers, footnote
# markers, formulas, figures/tables, footnotes and any stray bibliography.
NOISE_XPATH = (
    ".//tei:ref[@type='bibr'] | .//tei:ref[@type='foot'] | .//tei:formula"
    " | .//tei:figure | .//tei:note | .//tei:listBibl"
)
NOISE_TAGS = ("formula", "figure", "note", "listBibl")


class GrobidError(Exception):
    """Raised when GROBID cannot be reached or fails to process a PDF."""


def paper_id_and_year(pdf_path: Path) -> tuple[str, int]:
    """Get `(paper_id, year)` from a file name like `2024_000001.pdf`."""
    match = PAPER_ID_PATTERN.match(pdf_path.stem)
    if match is None:
        raise ValueError(f"Unexpected PDF file name: {pdf_path.name}")
    return pdf_path.stem, int(match["year"])


class GrobidParser:
    """Sends PDFs to a GROBID server and caches the resulting TEI files."""

    def __init__(
        self,
        server_url: str = GROBID_URL,
        cache_dir: Path = TEI_CACHE_DIR,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        force: bool = False,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.force = force
        self.reader = TeiReader()

    def parse(self, pdf_path: Path) -> Paper:
        """Parse one PDF into a Paper (without CSV metadata, labels or cleaning)."""
        paper_id, year = paper_id_and_year(pdf_path)
        return self.reader.read(self.get_tei(pdf_path), paper_id, year)

    def get_tei(self, pdf_path: Path) -> bytes:
        """Return the TEI for a PDF, from the cache unless `force` is set."""
        cache_path = self.cache_path(pdf_path)
        if cache_path.exists() and not self.force:
            return cache_path.read_bytes()
        tei = self._request_tei(pdf_path)
        _write_atomically(cache_path, tei)
        return tei

    def cache_path(self, pdf_path: Path) -> Path:
        paper_id, year = paper_id_and_year(pdf_path)
        return self.cache_dir / str(year) / f"{paper_id}.tei.xml"

    def _request_tei(self, pdf_path: Path) -> bytes:
        url = f"{self.server_url}/api/processFulltextDocument"
        data = {"consolidateHeader": "0", "consolidateCitations": "0"}
        try:
            with pdf_path.open("rb") as pdf_file:
                response = requests.post(
                    url, files={"input": pdf_file}, data=data, timeout=self.timeout
                )
        except requests.ConnectionError as error:
            raise GrobidError(
                f"GROBID is not reachable at {self.server_url}. Start it with:\n"
                "  docker run --rm --init --ulimit core=0 -p 8070:8070 "
                "grobid/grobid:0.9.1-crf"
            ) from error
        except requests.Timeout as error:
            raise GrobidError(
                f"GROBID timed out after {self.timeout}s on {pdf_path.name}"
            ) from error
        if response.status_code != 200:
            raise GrobidError(
                f"GROBID returned HTTP {response.status_code} for {pdf_path.name}: "
                f"{response.text[:200]}"
            )
        return response.content


class TeiReader:
    """Turns a GROBID TEI document into a Paper with abstract and sections."""

    def read(self, tei: bytes, paper_id: str, year: int) -> Paper:
        root = etree.fromstring(tei)
        return Paper(
            paper_id=paper_id,
            year=year,
            abstract=self._abstract(root),
            sections=self._sections(root),
        )

    def _abstract(self, root: etree._Element) -> str:
        abstract = root.find("tei:teiHeader/tei:profileDesc/tei:abstract", NS)
        if abstract is None:
            return ""
        # The abstract is wrapped as <abstract><div><p>...</p></div></abstract>.
        paragraphs = _without_noise(abstract).iter(f"{{{TEI_NS}}}p")
        blocks = [_normalize_space("".join(p.itertext())) for p in paragraphs]
        return "\n\n".join(block for block in blocks if block)

    def _sections(self, root: etree._Element) -> list[Section]:
        sections: list[Section] = []
        for location in ("body", "back"):
            container = root.find(f"tei:text/tei:{location}", NS)
            if container is None:
                continue
            for div, grobid_type in _content_divs(container):
                sections.append(self._section(div, location, grobid_type))
        return sections

    def _section(
        self, div: etree._Element, location: str, grobid_type: str | None
    ) -> Section:
        head = div.find("tei:head", NS)
        if head is not None:
            heading = _normalize_space("".join(head.itertext()))
            number = (head.get("n") or "").rstrip(".") or None
        else:
            heading = grobid_type or ""
            number = None
        return Section(
            heading=heading,
            number=number,
            location=location,
            text=_content_text(_without_noise(div)),
            grobid_type=grobid_type,
        )


def _content_divs(
    container: etree._Element, inherited_type: str | None = None
) -> Iterator[tuple[etree._Element, str | None]]:
    """Yield `(div, type)` for each div with its own content, in document order.

    GROBID wraps back-matter sections in a typed div without a head (e.g.
    `<div type="acknowledgement"><div><head>ACKNOWLEDGMENTS</head>...`). Such
    wrappers are skipped and their type is passed down to the inner divs.
    The bibliography (`div[@type="references"]`) is skipped entirely.
    """
    for div in container.findall("tei:div", NS):
        grobid_type = div.get("type") or inherited_type
        if grobid_type == "references":
            continue
        if _has_own_content(div):
            yield div, grobid_type
        yield from _content_divs(div, grobid_type)


def _has_own_content(div: etree._Element) -> bool:
    """True if the div has a head or any child other than nested divs and noise."""
    ignored = {"div", *NOISE_TAGS}
    return any(etree.QName(child).localname not in ignored for child in _element_children(div))


def _content_text(element: etree._Element) -> str:
    """Text of the element's children except `head` and nested `div`s, one block each."""
    blocks = []
    for child in _element_children(element):
        if etree.QName(child).localname in ("head", "div"):
            continue
        text = _normalize_space("".join(child.itertext()))
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)


def _without_noise(element: etree._Element) -> etree._Element:
    """Return a copy of the element with the NOISE_XPATH elements removed."""
    element = copy.deepcopy(element)
    for noise in element.xpath(NOISE_XPATH, namespaces=NS):
        _remove_keeping_tail(noise)
    return element


def _remove_keeping_tail(element: etree._Element) -> None:
    """Remove an element but keep the text that follows it (lxml drops it otherwise)."""
    parent = element.getparent()
    if parent is None:
        return
    tail = element.tail or ""
    previous = element.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail
    parent.remove(element)


def _element_children(element: etree._Element) -> list[etree._Element]:
    """Child elements, skipping comments and processing instructions."""
    return [child for child in element if isinstance(child.tag, str)]


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _write_atomically(path: Path, content: bytes) -> None:
    """Write via a temporary file so a crash or a parallel worker never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(content)
    os.replace(tmp_path, path)


def read_ids(ids_file: Path) -> list[str]:
    """Read one paper_id per line, skipping blank lines."""
    return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]


def pdf_path_for(pdf_dir: Path, paper_id: str) -> Path:
    """Path of a paper's PDF: `{pdf_dir}/{year}/{paper_id}.pdf`."""
    return pdf_dir / paper_id.split("_")[0] / f"{paper_id}.pdf"


def _print_debug(paper: Paper) -> None:
    print(f"\n=== {paper.paper_id}  abstract: {len(paper.abstract.split())} words, "
          f"sections: {len(paper.sections)}, total: {paper.n_words} words")
    for section in paper.sections:
        number = section.number or "-"
        grobid_type = f"  [{section.grobid_type}]" if section.grobid_type else ""
        print(f"  {number:>6}  {section.location:<4}  {section.heading}{grobid_type}"
              f"  ({len(section.text.split())} words)")


def main() -> None:
    """Debug command: print the abstract length and section list of each sample paper."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--ids-file", type=Path, default=Path("parsing/sample_ids.txt"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--force", action="store_true", help="ignore the TEI cache")
    args = parser.parse_args()

    grobid = GrobidParser(force=args.force)
    for paper_id in read_ids(args.ids_file):
        try:
            _print_debug(grobid.parse(pdf_path_for(args.pdf_dir, paper_id)))
        except (GrobidError, ValueError, OSError) as error:
            print(f"\n=== {paper_id}  ERROR: {error}")


if __name__ == "__main__":
    main()
