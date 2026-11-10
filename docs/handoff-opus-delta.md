# Hand-off: extra Opus 5.5 translations for a sensitivity of 0.1

## Goal

With three translations per condition, the smallest difference between two conditions that the
Opus 5.5 results resolve at the 95% level (the sensitivity Δ, `variance.py`) is 0.15–0.24 depending
on the film. At that Δ the paper cannot say whether the analysis condition beats unaided translation
or the bare meme list for Opus. More translations of three conditions bring Δ to about 0.1.

| Film | Translations per condition (total) | New per condition |
|---|---|---|
| ivan-vas | 10 | 7 |
| diamond-arm | 7 | 4 |
| gentelmen | 18 | 15 |
| carnival-night | 16 | 13 |
| pokrov-gate | 18 | 15 |

Conditions: `zero`, `list`, `list-analysis`; 162 new full-film translations in total. Targets
assume the variance pooled over conditions stays as measured; `variance.py` re-checks them at the end.

## Setup on the other machine

1. Clone the repository, `pip install -r requirements.txt`.
2. Claude Code logged in with the subscription: translations (Opus 5.5) and evaluations
   (Sonnet 5.5) run through `claude -p` (`code/claude_cli.py`). The wrapper pauses at 80% of the
   5-hour limit by day and 95% at night; long gaps are normal.
3. OpenAI key in `./LYS-API-key.txt` (repository root, not committed): the span extractor uses
   GPT-6 Sol and the span verifier GPT-5.4.

## Running

Per film, from the repository root, in this order:

```
code/run_opus_delta.sh <film> translate   # adds runs 4..N for the three conditions
code/run_opus_delta.sh <film> map         # time-code spans + verification
code/run_opus_delta.sh <film> eval        # Sonnet 5.5, Meme-MQM v2, 2 runs, new runs only
code/run_opus_delta.sh <film> stats       # aggregate, then variance.py at 0.1
```

Configs: `yaml-pipelines/films/<film>-russian-english-opus-delta.yaml`. `translate` skips runs that
exist, so an interrupted phase can be restarted. Start with diamond-arm (fewest runs) to check the
whole chain, then the others; films can run in parallel terminals, but they share one rate limit.

- `map` ends with the verifier's `FLAGGED` line. Flagged new runs: re-locate by content with
  `extract_meme_spans.py ... --runs method:run,... --content` (printed by the script), then run
  `map` again. Truncated translations keep their "ERROR" spans; they count as omissions.
- `eval` must not re-score runs 1–3: the evaluator writes a second file next to an existing one,
  which would add evaluation runs to them. The script passes only unscored runs above 3.
- `stats` writes `merged_summary.json` (read by `paper_tables.py`) and `method_comparison.json`
  with each condition's sensitivity. If conditions still miss 0.1, the summary names the required
  number of translations; raise `n_runs` in the config and repeat.

Commit after each film: translations, `claude-opus-5-5.json`, the `mapping/` logs, and the
`mqm-memes-v2` evaluation folder.

## Not covered

The comparison evaluators (GPT-5-mini, Haiku 5.5), neural metrics and the whole-text judge are not
run on the new translations. The tables that use them for Opus compare the translations the human
scored (runs 1–3) or are diagnostics; the judges table (Sonnet vs GPT-5-mini condition means) would
compare different sets of translations unless GPT-5-mini is also run on the new ones.

## Back in the paper

Regenerate the tables (`python code/paper_tables.py --paper <paper repo>`), show each film's Δ in the
first column of the results tables, and check every claim that compares conditions against it.
