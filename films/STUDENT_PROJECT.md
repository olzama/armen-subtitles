# Research Project: What Makes Cultural Translation and Translation Evaluation Hard for LLMs?

## Overview

This project studies how reliably large language models (LLMs) translate culturally-loaded content — jokes, memes, idiomatic humor, cultural allusions — and how reliably other LLMs can *evaluate* those translations. The infrastructure for running translation experiments and measuring reliability is already built (see `README.md`). Your contribution will be:

1. Building a new dataset of an English film rich in cultural content, with translations into Spanish (or Galician)
2. Running it through the existing pipeline
3. Analyzing *what kinds of content* make translation or evaluation unreliable

The goal is a TACL paper which you would be able to present at ACL -- but in this case, Olga will be first author. Which means she will also write most of the paper etc. If you would rather be first authors, we can organize that too. Then you would 


## The core research question, in plain terms

When you ask an LLM to translate a culturally-loaded line multiple times, you get slightly different results each time. Sometimes the variation is small — the model clearly "knows" what to do. Other times the variation is large — different runs produce radically different translations. We call this **translation variance** (σ_T in the code).

Similarly, when you ask an LLM to *judge* the quality of a translation multiple times, the scores vary. A clearly bad translation gets consistently low scores; a genuinely ambiguous one gets scattered scores. We call this **evaluator variance** (σ_E).

The interesting question is: **do the same items drive both kinds of variance, or different ones?** An item might be hard to translate consistently (high σ_T) but easy to evaluate (the evaluator always agrees it's good or bad). Or it might be easy to translate but hard to evaluate (was this pun preserved? the evaluator isn't sure). Understanding this distinction tells us something deep about where LLMs succeed and fail with culturally-specific language.

An existing experiment — translating a Soviet comedy (*Ivan Vasilievich Changes Profession*, 1973) from Russian into Galician — provides a reference point. Your dataset will let you ask whether the patterns you find are specific to that language pair and cultural context, or more general.

---

## Phase 1 — Build the dataset

### Choose a film

Pick an English-language film that is **dense with culturally-loaded language**: jokes that rely on shared cultural knowledge, memes, wordplay, register-based humor, pop culture references, absurdist humor. Good candidates:

- *Shrek* (fairy-tale deconstruction, pop culture parody, multi-register humor)
- *The Big Lebowski* (Coen Brothers deadpan, cultural archetypes)
- *Monty Python and the Holy Grail* (absurdist humor, historical parody)
- *Anchorman* (character-based humor, 70s cultural references)
- *Mean Girls* (social register humor, teen culture references that date very specifically)

The film should be one you know well — you are the linguistic expert here. Download or obtain the English subtitle file (SRT format).

### Identify the challenging units (using e.g. https://en.wikiquote.org/)

Not every line is interesting. Use some "objective" source (such as Wikiquote) to pick the lines which have become memes/culturally significant. They may be easy or hard to translate, that depends. It is certain that some of them will be hard to translate so that the effect is preserved.



### Write the cultural analysis (in collaboration with an LLM)

For each selected unit, you need to obtain/write a short analysis. When I did that for Russian films, I asked ChatGPT to do it and only corrected it where it was clearly off/clueless. Ask it to output everything strictly in the following `reference.json` format:

```json
{
  "id": 1,
  "character": "Character Name",
  "original": { "eng": "The original English line" },
  "reference": {
    "spa": "Your Spanish reference translation"
  },
  "analysis": {
    "general": {
      "text": "What makes this line culturally loaded and what a good translation needs to preserve.",
      "nb": "Any specific pitfall or mistranslation to watch out for (optional)."
    },
    "language_specific": {
      "spa": {
        "text": "Any Spanish-specific translation note (optional).",
        "nb": null
      }
    }
  },
  "segment_number": [123]
}
```

The `segment_number` field refers to the subtitle segment number in the SRT file — needed for the automatic mapping step later (see `README.md`, section *Compiling data for evaluation*).

All of this involves some manual work, but not too much. ChatGPT/Gemini should be able to do most of it.


### Work with ChatGPT/Gemini to produce reference translations

Ask a good LLM (in the chat mode) to translate each selected unit into Spanish and/or Galician. These are your **reference translations** — not necessarily perfect, but your best judgment of what a culturally faithful translation looks like. The pipeline will use them as the evaluation baseline.

At the end of this phase you should have a `reference.json` file for your film, a subtitle SRT file, and the context files (a character sheet, a short plot summary) following the structure in `films/data/ivan-vas/summaries/`.

---

## Phase 2 — Run the pipeline

The pipeline is already built. Use one of the existing experiment configs as a starting point:

```
yaml-pipelines/films/ivan-vas-russian-galician.yaml
```

Ask an LLM to edit it for your film, source/target languages, and your data files. The `README.md` (section *Pipeline driver*) explains each field.

Then run:

```
python run_pipeline.py yaml-pipelines/films/my-film-english-spanish.yaml
```

The pipeline will translate your selected units using multiple prompting strategies (`zero`, `characters`, `list-analysis`, etc.), evaluate each translation multiple times using an LLM judge, and compute reliability statistics. Check status at any time with `--status`. The interactive mapping step (`map_translation_segments.py`) aligns the translated subtitle files to your `reference.json` items — this requires human review but is fast once you get the hang of it.

After the pipeline finishes, run:

```
python code/variance.py my-film gpt-5.2 gpt-5.4-mini 0.05 English Spanish mqm-memes
```

This will print a table showing, for each translation method, the current sensitivity and whether more translation or evaluation runs are needed. Aim for `variance_delta: 0.05` — meaning the pipeline has enough data to reliably detect a difference of 0.05 between two methods.

---

## Phase 3 — Extract per-item variance

The existing pipeline computes σ_T and σ_E *per method* (averaged across all items). For this research you need them *per item*. Write a script that reads the raw evaluation files from:

```
films/output/eval/llm-eval/<film>/<lang-pair>/<model-combo>/<prompt>/
```

Each file `run_N_eval_M.json` contains per-item MQM judgments. From these, compute:

- **σ_E per item**: for a given translation run, how much do the evaluation scores vary across eval runs?
- **σ_T per item**: across all translation runs of the same method, how much do the *mean* scores (averaged across eval runs) vary?

Do this for each method separately, and also averaged across methods. The result is a table: one row per item, with columns for σ_T, σ_E, mean score, and method.

Look at `code/variance.py` and `code/aggregate_mqm.py` for how the existing per-method computation works — the per-item version follows the same logic.

---

## Phase 4 — Analysis (≈ 3 weeks)

With σ_T and σ_E per item, you can now ask the research questions.

### 4a. Do the same items drive both kinds of variance?

Plot σ_T against σ_E across items. If they are correlated, the same content is hard for both the translator and the evaluator. If they are not, the two kinds of variance are capturing different things — which is the more interesting finding.

### 4b. Does cultural content type predict variance?

Add a column to your item table for the cultural type you assigned in Phase 1 (allusion / register / wordplay / meme). Test whether different types systematically produce higher σ_T or σ_E. For example:

- Wordplay items might have high σ_T (many different solutions) but low σ_E (it is obvious when the pun is lost)
- Allusion items might have low σ_T (one obvious strategy: keep the reference or drop it) but high σ_E (evaluators disagree on whether the cultural knowledge transfers)

### 4c. Does method type affect variance on specific content types?

Compare σ_T for the same item across methods. Does providing cultural context (`list-analysis`) reduce σ_T more on allusion items than on wordplay items? Does providing a reference translation (`given`) effectively collapse σ_T for all content types?

### 4d. Cross-dataset comparison

The Russian→Galician dataset (Soviet-era film, 1973) is available in `films/data/ivan-vas/`. Repeat the analysis on that dataset and compare. Do the same cultural content types produce high variance regardless of language pair and cultural source? Or is the pattern specific to the cultural context?

---

## Phase 5 — Write up (≈ 2 weeks)

The framing for the paper: **σ_T and σ_E as complementary diagnostics of LLM reliability on culturally-loaded translation**.

The contribution is threefold:
- A new benchmark dataset of culturally-annotated English film subtitles with Spanish reference translations
- An analysis method for decomposing translation reliability at item level
- Empirical findings on which types of cultural content drive each kind of variance, with cross-language-pair comparison

Target venues: workshops on cultural NLP, translation quality estimation, or LLM evaluation at ACL or EMNLP.

---

## What to avoid

- **Do not collect new human translations or evaluations** — the existing automated pipeline produces enough data for the analysis, and human data collection would take the full two months on its own.
- **Do not over-design the annotation scheme** — two or three cultural content categories are enough. The goal is signal, not a complete typology.
- **Do not try to beat the translation quality** — the research question is about reliability, not about finding the best translation method.
