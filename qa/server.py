"""Local web page to label the AI-disclosure hits, one paper at a time.

The label and note of a paper are written to every row of that paper in the CSV.

Usage: docker compose up -d qa, then open http://localhost:8000 (stop: docker compose stop qa)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from analysis.disclosure import COLUMNS, PATTERNS

LABELS = ("used_ai", "no_ai_statement", "not_disclosure", "unsure", "")  # "" = not labeled yet
PAGE = Path(__file__).with_name("index.html")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path} has old columns. Re-run analysis.disclosure (with --force).")
        return list(reader)


def papers(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Rows grouped by paper, in file order. Each hit has the character spans of its matches."""
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        paper = grouped.setdefault(row["paper_id"], {
            "paper_id": row["paper_id"], "year": row["year"], "title": row["title"],
            "label": row["label"], "note": row["note"], "hits": [],
        })
        sentence = row["sentence"]
        spans = sorted(m.span() for p in PATTERNS.values() for m in p.finditer(sentence))
        paper["hits"].append({
            "section": row["section"], "section_labels": row["section_labels"],
            "terms": row["terms"], "sentence": sentence, "spans": spans,
        })
    return list(grouped.values())


def save_label(path: Path, paper_id: str, label: str, note: str) -> None:
    """Set the label and note of every row of one paper, then replace the file atomically."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    rows = load(path)
    matched = [row for row in rows if row["paper_id"] == paper_id]
    if not matched:
        raise ValueError(f"unknown paper {paper_id!r}")
    for row in matched:
        row["label"], row["note"] = label, note
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)  # a crash never leaves half a file


def make_handler(csv_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/papers":
                self._send_json(200, {"labels": LABELS[:-1], "papers": papers(load(csv_path))})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/label":
                return self._send_json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                save_label(csv_path, str(body["paper_id"]), str(body["label"]), str(body.get("note", "")))
            except (ValueError, KeyError, TypeError) as error:
                return self._send_json(400, {"error": str(error)})
            self._send_json(200, {"ok": True})

        def _send_json(self, status: int, data: object) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Label the AI-disclosure hits in a browser.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load(args.csv)  # fail early on a wrong file
    # 0.0.0.0 so Docker can forward the port; compose.yaml publishes it on localhost only.
    server = HTTPServer(("0.0.0.0", args.port), make_handler(args.csv))
    print(f"Open http://localhost:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
