# Hand-off: neural metrics on the meme passages, and the whole-text claim

## Goal

The metric-agreement tables of the paper (`tables/metrics.tex`, `tables/metrics-gpt52.tex`) correlate
each automatic metric's ranking of the translation conditions with the human Meme-MQM ranking. The
human scored only the meme passages; CometKiwi, xCOMET and MetricX score the whole subtitle file.
The paper attributes their disagreement to dilution ("a whole-text score averages them away"), which
is not tested: the metrics could also fail on the memes themselves.

Two jobs:

1. **Paper wording.** Limit the claim to what the tables show, and state that the metrics receive
   less information than Meme-MQM.
2. **Meme-only neural metrics.** Score the same three metrics on the meme passages alone and add
   them to the agreement tables. This separates dilution from inability.

Job 2 decides the final wording of job 1, so run job 2 first.

## Job 2: meme-only neural metrics

### What to score

The Meme-MQM evaluation files already hold, for every translation, each meme's source text and the
span that renders it:

```
experiments/films/output/eval/llm-eval/<film>/Russian-<tgt>/<translator>-by-claude-sonnet-5-5/mqm-memes-v2/<method>/run_<r>_eval_1.json
  items[]: id, source, candidate, reference, ...
```

Score each `(source, candidate)` pair reference-free, with the three models of
`code/evaluate_neural_metrics.py` (`CometScorer`, `MetricXScorer`; reuse them). These are the units
the human and the Meme-MQM evaluators judged, so no subtitle alignment is needed. `eval_1` and
`eval_2` of a run hold the same spans; read one.

- Translators: `gpt-6-sol`, `claude-opus-5-5`, `gpt-5.2`.
- Films: the five in `FILMS` of `code/paper_tables.py`, Russian→English; for `gpt-5.2` also
  `ivan-vas` Russian→Galician.
- Runs: every run that has an evaluation file. The human-judged runs of the current translators
  are 1, 2 and 3, and the tables need exactly those.
- Skip items with an empty `candidate` and count them in the output.

### Output

New script `code/evaluate_neural_metrics_memes.py`, one file per film, language pair, translator
and metric, next to the whole-file scores:

```
experiments/films/output/eval/comet-eval/<film>/Russian-<tgt>/<translator>/meme_scores_<metric>.json
  {"metric", "model", "higher_is_better", "unit": "meme span",
   "methods": {<method>: {"runs": {<r>: {"mean": float, "n_items": int, "n_skipped": int,
                                          "items": {<meme id>: float}}}}}}
```

Runs are keyed by their run number, not by position (the whole-file files use a list, which
`paper_tables.py` indexes as run `i + 1`).

### Tables

In `code/paper_tables.py`:

- `table_metrics`: add three columns, "Memes only: CometKiwi, xCOMET, MetricX", between the
  whole-text and the Meme-MQM columns. Same computation as the existing metric columns (sign from
  `METRIC_COLS`; for the current translators the score of the human-judged run, for GPT-5.2 the
  condition mean over runs). The `Mean`, `$p$` and `no references` rows then cover them without
  further code. Update both captions.
- `table_meme_level`: add the three metrics as columns. The per-item scores correlate directly with
  the human per-item scores from `human_items()`; orient MetricX with a minus sign.

Regenerate with `python code/paper_tables.py --paper <paper checkout>`; all other tables must come
out unchanged.

### Setup

- GPU. The workload is small: about 750 translations of 8 to 23 short pairs each, per metric.
- Hugging Face login with access to the gated COMET models (`hf_login()` in the existing script).
- MetricX code at `~/src/metricx` (`git clone https://github.com/google-research/metricx`).

## Job 1: paper wording

Paper repository: `film-subtitles-paper`, file `main_lre.tex`. Four places make the claim:

| Place | Current wording |
|---|---|
| Abstract, last sentence | "tracks human judgment far more closely than neural metrics or a whole-text LLM judge" |
| Evaluator validation, first paragraph | "The whole-text metrics do not agree: …"; "a whole-text score averages them away" |
| Evaluator validation, GPT-5.2 paragraph | "that of the whole-text metrics is not" |
| Conclusion, second paragraph | "do not agree with it, because the few passages on which translations differ are averaged in with thousands on which they do not" |

Changes, independent of the outcome of job 2:

- Say what is compared: the human judged the meme passages only, so the tables show whether a
  metric tracks the meme-level differences between conditions, not whether it agrees with a human
  judgment of the whole translation.
- State the asymmetry once, in the evaluator-validation section: the neural metrics run
  reference-free, while Meme-MQM is given the approved reference and analysis; the comparison is
  between evaluation with and without that human input.

Changes that depend on job 2:

| Meme-only metrics | Wording |
|---|---|
| Still do not agree with the human | The failure is not only dilution: on the meme passages themselves the metrics do not track the human. Keep "averages them away" for the whole-text columns and add this as the second reason, next to the existing remark that the whole-text judge applies the wrong standard. |
| Agree with the human | Dilution is confirmed: the metrics see the differences once the memes are isolated. The contribution of the protocol is then the selection of the passages, and the text should say so; compare their agreement with that of the Meme-MQM evaluators in the same table. |

The paper states numbers only in tables; the prose describes what a table shows.

## Done when

- `meme_scores_<metric>.json` exists for the three metrics, three translators and all films above.
- `tables/metrics.tex`, `tables/metrics-gpt52.tex` and `tables/meme-level.tex` carry the meme-only
  columns; no other table changed.
- The four places in `main_lre.tex` are reworded, and the paper compiles.
