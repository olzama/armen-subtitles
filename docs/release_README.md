# Soviet comedy memes: a dataset for evaluating the translation of culturally loaded passages

This repository accompanies the paper *Hard to Translate, Harder to Detect: Human Analysis Guides LLMs in Meme Translation---A Dataset and Validated Evaluation* (submitted to *Language Resources and Evaluation*). It contains a reception-based dataset of famous quotations (memes) from five Soviet film comedies, the human-validated analyses and reference translations built for them, the meme passages of full-film LLM subtitle translations under several prompting conditions, and every automatic and human evaluation reported in the paper. The films are under copyright, so only the meme passages are distributed: the source subtitles are identified by their OpenSubtitles.org file IDs, and the full-film translations are available from the authors on request.

## Contents

| Path | What it is |
|---|---|
| `data/<film>/reference.json` | The audience-selected memes: Russian text, position in the source subtitle file (`segment_number`), human-validated analysis, and approved reference translations (English; Spanish/Galician where available). |
| `data/<film>/meme-list.json`, `list-analysis-*.json` | The meme lists and analyses as given to the translators in the *meme list* and *list+analysis* conditions. |
| `data/<film>/summaries/` | Film-level context used in the prompts (character descriptions, plot summary, etc.). |
| `data/ivan-vas/hil/` | Logs of the human-in-the-loop construction step for *Ivan Vasilievich*: the conversation with the LLM in which the analyses and English references were developed, and the Galician evaluator's notes behind the Galician language addendum. |
| `sources.json` | The Russian source subtitle files: OpenSubtitles.org file ID, IMDb ID, SHA-256 checksums of the download and of the file used, entry count, and the recipe that rebuilds the file used from the download. |
| `prompts/` | All translation and evaluation prompts. |
| `translations/<film>/<translator>.memes.json` | Each meme passage as rendered in each translation (what the evaluators scored), by target language, condition, and translation number. |
| `translations/<film>/Russian-<Lang>/<translator>/meme-search/meme-analysis/meme-analysis.txt` | The translator's own meme list and analysis (*meme-search* condition). |
| `eval/llm-eval/<film>/Russian-<Lang>/<translator>-by-<evaluator>/mqm-memes-v2/` | Every Meme-MQM evaluation run with the revised evaluation prompt (`prompts/eval/mqm-memes-v2.txt`), per meme: candidate, issues, severities; and the aggregated summaries with variance components. These are the scores the paper reports for the Claude evaluators: Claude Sonnet 5.5 (main) and Claude Haiku 5.5. |
| `eval/llm-eval/<film>/Russian-<Lang>/<translator>-by-<evaluator>/mqm-memes/` | The same with the original evaluation prompt (`prompts/eval/mqm-memes.txt`): the GPT comparison evaluators (GPT-5-mini; GPT-5.4-mini for Galician and Spanish), whose scores the paper reports, and the earlier Claude runs, kept for comparison. The original prompt penalizes renderings that replace a name or reference in the analysis with one that does the same job for the target audience; the revised prompt does not. |
| `eval/llm-eval/.../mqm-wholefile/` | The whole-text LLM judge (GPT-5.4 for GPT-5.2, Claude Sonnet 5.5 for the current translators): every issue with its severity and category; the quoted dialogue is kept only for issues that concern a meme. |
| `eval/human-eval/` | Human Meme-MQM judgments (one evaluation session per film and evaluator) and their summaries. `evaluator-en` is the English evaluator, `evaluator-gl` the Galician one; evaluator profiles are not released. |
| `eval/comet-eval/` | Time-code-aligned CometKiwi, xCOMET and MetricX scores. |
| `eval/difficulty/` | Translation-difficulty signals per subtitle entry, and the meme-search coverage judgments (`meme_search_coverage-by-claude-sonnet-5-5.json`). |
| `translations/<film>/gpt-5.2-humanspans.json` | The GPT-5.2 translations scored by the human evaluator, at the meme spans the evaluator saw (before the uniform span mapping), with their Sonnet scores under `eval/llm-eval/.../gpt-5.2-humanspans-by-claude-sonnet-5-5/`. |
| `code/`, `configs/`, `run_pipeline.py` | Code and pipeline configurations for every experiment. |

Films: *Ivan Vasilievich Changes Profession* (`ivan-vas`), *The Diamond Arm* (`diamond-arm`), *Gentlemen of Fortune* (`gentelmen`), *Carnival Night* (`carnival-night`), *The Pokrov Gates* (`pokrov-gate`). Translators: GPT-6 Sol, Claude Opus 5.5, GPT-5.2, GPT-5.4-mini.

## Source subtitles

The Russian source subtitles are not redistributed. Segment numbers in the dataset refer to the exact files listed in `sources.json`, which differ from the OpenSubtitles.org downloads only in encoding, line endings, and a few blank lines (and, for *Carnival Night*, entries numbered from 0). To retrieve them, rebuild the exact files, and verify both checksums (requires a free OpenSubtitles.org account and API key):

```
OPENSUBTITLES_API_KEY=... OPENSUBTITLES_USERNAME=... OPENSUBTITLES_PASSWORD=... python code/fetch_sources.py --sources sources.json --out data
```


## Reproducing the paper's tables

```
pip install -r requirements.txt
python code/paper_tables.py --paper <dir>     # writes <dir>/tables/*.tex from eval/
```

The whole-film metrics (CometKiwi, xCOMET, MetricX, and the whole-text judge) cannot be recomputed from this release alone, since it does not contain the full translations; their scores are included. Re-running translation or evaluation requires API access to the models named in `configs/` (see `code/translate.py`, `code/evaluate_mqm_parallel.py`, `code/claude_cli.py`).

## Licenses

Code (`code/`, `configs/`, `run_pipeline.py`): MIT, see `LICENSE`. Data: CC BY 4.0, see `LICENSE-DATA.md`. The Russian meme passages and their translations are quotations from copyrighted films; they are included for research and analysis and are not covered by either license.

## Citation

Please cite the dataset as archived on Zenodo: https://doi.org/10.5281/zenodo.23504282 (this DOI always resolves to the latest version; each version also has its own DOI). The paper's citation will be added on publication.
