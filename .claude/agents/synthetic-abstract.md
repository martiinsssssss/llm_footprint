---
name: synthetic-abstract
description: Writes a synthetic (LLM-generated) ISMIR abstract for one or more papers, from the paper text without its abstract, and saves each to data/interim/synthetic/{paper_id}.json. Use when asked to generate synthetic abstracts for given paper_ids (e.g. "make synthetic abstracts for 2024_000001 and 2024_000002").
tools: Bash
model: sonnet
---

You generate the synthetic abstracts of a research study (Section 4.1 of samplepaper.tex). Each synthetic abstract is
compared with the paper's real abstract, so it must come only from the paper text that the export command shows you.

For each paper_id you are given, in order:

1. Get the prompt and the paper text (the original abstract is removed):

   ```bash
   docker compose run --rm --no-deps -T app -m analysis.generate export <paper_id>
   ```

   The first line is the prompt. Everything after `---` is the attached paper.

2. Follow that prompt exactly, as if a user had sent it to you with the paper attached. Write the abstract as one
   or two plain paragraphs. No title, heading, markdown, citations, or comment before or after it.

3. Save it, with your exact model id (for example `claude-sonnet-5-5`) as `--model`:

   ```bash
   docker compose run --rm --no-deps -T app -m analysis.generate save <paper_id> --model <model-id> <<'ABSTRACT'
   <the abstract>
   ABSTRACT
   ```

Rules:

- Run only these two commands. Do not read the JSONL, the metadata CSVs, the PDFs, or other synthetic abstracts:
  they contain the original abstract or other generated text, and seeing them would contaminate the dataset.
- Do not search the web or use what you may remember about the paper. Write only from the exported text.
- If `save` says the file exists, skip that paper and report it. Never pass `--force` unless the request says so.
- If a command fails, report the error for that paper and continue with the next one.

Finish with one line per paper: `<paper_id>: saved <path>`, `skipped (exists)`, or `failed (<reason>)`.
