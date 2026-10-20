# Soviet comedy memes: a dataset for evaluating the translation of culturally loaded passages

This repository accompanies the paper *Hard to Translate, Harder to Detect: Human Analysis Guides LLMs in Meme Translation---A Dataset and Validated Evaluation* (submitted to *Language Resources and Evaluation*). It contains a reception-based dataset of famous quotations (memes) from five Soviet film comedies, the human-validated analyses and reference translations built for them, full-film LLM subtitle translations under several prompting conditions, and every automatic and human evaluation reported in the paper.

## Contents

| Path | What it is |
|---|---|
| `data/<film>/reference.json` | The audience-selected memes: Russian text, position in the source subtitle file (`segment_number`), human-validated analysis, and approved reference translations (English; Spanish/Galician where available). |
| `data/<film>/meme-list.json`, `list-analysis-*.json` | The meme lists and analyses as given to the translators in the *meme list* and *list+analysis* conditions. |
| `data/<film>/summaries/` | Film-level context used in the prompts (character descriptions, plot summary, etc.). |
| `sources.csv` | The Russian source subtitle files: film, file name, SHA-256 checksum, entry count, OpenSubtitles.org file ID. |
| `prompts/` | All translation and evaluation prompts. |
| `translations/<film>/Russian-<Lang>/<translator>/<condition>/translation-<n>.srt` | Full-film translations. |
| `translations/<film>/<translator>.memes.json` | The meme span located in each translation (what the evaluator scored). |
| `translations/<film>/Russian-<Lang>/<translator>/meme-search/meme-analysis/` | The translator's own web search and meme analysis (*meme-search* condition). |
| `eval/llm-eval/<film>/Russian-<Lang>/<translator>-by-<evaluator>/mqm-memes/` | Every Meme-MQM evaluation run (per meme: candidate, issues, severities) and the aggregated summaries with variance components. Main evaluator: Claude Sonnet 5.5; comparison evaluators: GPT-5-mini (GPT-5.4-mini for Galician and Spanish) and Claude Haiku 5.5. |
| `eval/llm-eval/.../mqm-wholefile/` | The whole-text LLM judge (GPT-5.4 for GPT-5.2, Claude Sonnet 5.5 for the current translators). |
| `eval/human-eval/` | Human Meme-MQM judgments (one evaluation session per film and evaluator) and their summaries. `evaluator-en` is the English evaluator, `evaluator-gl` the Galician one; evaluator profiles are not released. |
| `eval/comet-eval/` | Time-code-aligned CometKiwi, xCOMET and MetricX scores. |
| `eval/difficulty/` | Translation-difficulty signals per subtitle entry, and the meme-search coverage judgments (`meme_search_coverage-by-claude-sonnet-5-5.json`). |
| `translations/<film>/gpt-5.2-humanspans.json` | The GPT-5.2 translations scored by the human evaluator, at the meme spans the evaluator saw (before the uniform span mapping), with their Sonnet scores under `eval/llm-eval/.../gpt-5.2-humanspans-by-claude-sonnet-5-5/`. |
| `code/`, `configs/`, `run_pipeline.py` | Code and pipeline configurations for every experiment. |

Films: *Ivan Vasilievich Changes Profession* (`ivan-vas`), *The Diamond Arm* (`diamond-arm`), *Gentlemen of Fortune* (`gentelmen`), *Carnival Night* (`carnival-night`), *The Pokrov Gates* (`pokrov-gate`). Translators: GPT-6 Sol, Claude Opus 5.5, GPT-5.2, GPT-5.4-mini.

## Source subtitles

The Russian source subtitles are not redistributed. Segment numbers in the dataset refer to the exact files listed in `sources.csv`. To retrieve and verify them:

```
OPENSUBTITLES_API_KEY=... python code/fetch_sources.py --csv sources.csv --out data
```

The script downloads each file and checks its SHA-256 against `sources.csv`.

## Reproducing the paper's tables

```
pip install -r requirements.txt
python code/paper_tables.py --paper <dir>     # writes <dir>/tables/*.tex from eval/
```

Re-running translation or evaluation requires API access to the models named in `configs/` (see `code/translate.py`, `code/evaluate_mqm_parallel.py`, `code/claude_cli.py`).

## Licenses

Code: MIT. Data, analyses, reference translations, and model outputs: CC BY 4.0. *(to be confirmed by the authors)*

## Citation

*(to be added on publication)*
