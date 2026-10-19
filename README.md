# LLM Footprint

## Scrapers

Each scraper lives in `scrapers/` and runs in one shared Docker image.

### ISMIR

Downloads ISMIR papers (2020–2025) and saves metadata plus the abstract, discussion and conclusion of each paper to `<out>/metadata_<year>.csv` (one file per year; re-running a year replaces only that file). PDFs go to `<out>/pdfs/<year>/`. Change `$(pwd)/data` to pick the output folder.

```bash
docker build -t scrapers .                                                                  # build the image
docker run --rm -v "$(pwd)/data:/data" scrapers scrapers/ismir.py --out /data                     # all years
docker run --rm -v "$(pwd)/data:/data" scrapers scrapers/ismir.py --out /data --years 2024 2025   # some years
docker run --rm -v "$(pwd)/data:/data" scrapers scrapers/ismir.py --out /data --limit 3           # quick test
```
