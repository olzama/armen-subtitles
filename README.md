## AI assisted creative translation and translation evaluation
The project consists of code which translates movie subtitles and evaluates the translations. It can easily be adapted to other domains, but is meant specifically for creative, highly challenging domains which do not yet allow fully automatic translation.

The code makes calls to AI APIs such as OpenAI and Gemini. You can easily adapt it to any AI API to which you have access.

```
pip install -r requirements.txt
```

### Directory structure

All film data and outputs live under `films/`. Experiment configs live under `experiments/`.

```
experiments/
  <film>-<src>-<tgt>.yaml    ← one config file per experiment (see Pipeline driver below)

films/
  data/<film_name>/
    subs/                    ← source SRT file(s)
    reference.json           ← reference translations of challenging units
    summaries/               ← optional context files (summaries, character sheets, etc.)
  prompts/
    eval/                    ← evaluation prompt files
    lang/                    ← optional language-specific translation instructions
  output/
    translations/<film_name>/<src_lang>-<tgt_lang>/<trans_model>/<method>/  ← raw translation runs
    translations/<film_name>/<trans_model>.json                             ← mapped translations (for evaluation)
    eval/llm-eval/<film_name>/<src_lang>-<tgt_lang>/<trans_model>-by-<eval_model>/  ← evaluation results
```

### Pipeline driver

The pipeline has several steps that must be run in order, and it is easy to lose track of what has been done. The recommended approach is to define each experiment in a YAML config file and use `run_pipeline.py` to manage it.

**Create a config** by copying `experiments/ivan-vas-russian-galician.yaml` and editing the fields:

```yaml
film: my-film
source_lang: Russian
target_lang: English
trans_model: gpt-5.2
eval_model: gpt-5.4-mini
temperature: 0.8
eval_runs: 4
eval_prompt: films/prompts/eval/mqm-memes.txt
variance_delta: 0.05

methods:
  - name: zero
    n_runs: 6
  - name: list
    n_runs: 3
    prompt: films/prompts/unit-list.txt
    unit_list: films/data/my-film/meme-list.json
  - name: list-lang
    n_runs: 3
    prompt: films/prompts/unit-list.txt
    unit_list: films/data/my-film/meme-list.json
    lang_prompt: true   # appends films/prompts/lang/<target_lang>.txt
  # ... add more methods as needed
```

**Check status** at any time to see what is done and what is missing:
```
python run_pipeline.py experiments/my-experiment.yaml --status
```

**Run individual steps:**
```
python run_pipeline.py experiments/my-experiment.yaml --step translate
python run_pipeline.py experiments/my-experiment.yaml --step eval
python run_pipeline.py experiments/my-experiment.yaml --step aggregate
python run_pipeline.py experiments/my-experiment.yaml --step variance
```

**Run all steps** (the script will pause at the interactive mapping step and print the command to run):
```
python run_pipeline.py experiments/my-experiment.yaml
```

Translation runs that already exist are skipped automatically. Evaluation will refuse to run if the mapped JSON is missing.

Note: the mapping step is always interactive and must be run manually (see [Compiling data for evaluation](#compiling-data-for-evaluation) below).

### Translating

Generic form:
```
python code/translate.py <film_name> <method> <trans_model> <temperature> <n_runs> <source_lang> <target_lang> [--prompt FILE] [--summary FILE] [--unit_list FILE] [--given_trans FILE] [--start-num N]
```

`method` is a free label for the translation approach (e.g. `zero`, `summary`, `given`). It has no effect on the translation itself — it names the output subdirectory, which is then used in evaluation.

**Zero-shot (no additional context):**
```
python code/translate.py sample-ivan-vas zero gpt-5.2 0.8 6 Russian English
```
Produces 6 translations under `films/output/translations/sample-ivan-vas/Russian-English/gpt-5.2/zero/`.

**With a summary or character sheet:**
Useful for experimenting with prompts that provide background context.
```
python code/translate.py sample-ivan-vas characters gpt-5.2 0.8 3 Russian English --prompt films/prompts/characters.txt --summary films/data/sample-ivan-vas/summaries/characters.txt
```

**With a list of challenging units:**
Asks the model to pay special attention to culturally or linguistically difficult items identified in advance.
```
python code/translate.py sample-ivan-vas list gpt-5.2 0.8 3 Russian English --prompt films/prompts/unit-list.txt --unit_list films/data/sample-ivan-vas/meme-list.json
```

**With pre-approved translations of challenging units:**
Integrates previously approved translations of difficult items into the full translation. A good way to enforce consistency on items that need human oversight.
```
python code/translate.py sample-ivan-vas given gpt-5.2 0.8 3 Russian English --prompt films/prompts/given-translations.txt --given_trans films/data/sample-ivan-vas/reference.json
```

### Compiling data for evaluation

In order to save on API calls, we suggest you only evaluate the challenging units. The rest is very likely to be translated well anyway.

To evaluate economically against a reference, you need to first extract the translation portions corresponding to the challenging units. If you want to evaluate without a reference, we suggest you evaluate translations in full to get better results.

If you work with SRT subtitles with numbered segments and have a reference JSON where those segment numbers are indicated, you can use the interactive mapping script to align the translations to the original items:
```
python code/map_translation_segments.py <film_name> <trans_model> <source_lang> <target_lang>
```

For example:
```
python code/map_translation_segments.py sample-ivan-vas gpt-5.2 Russian English
```

This reads the translated SRT files from `films/output/translations/sample-ivan-vas/Russian-English/gpt-5.2/` and writes the mapped translations to `films/output/translations/sample-ivan-vas/gpt-5.2.json`.

The script is interactive. For each item it proposes a segment mapping by searching for the translation segment most semantically similar to the reference translation (or to a previously accepted mapping for that item, if one exists). You review items in batches:
- Press **ENTER** to approve all items in a batch, or enter numbers (e.g. `2 5`) to flag specific items for correction.
- For a flagged item: press **ENTER** to accept the current proposal, **n** to widen the search window and get a new proposal, type segment number(s) to pick different segments and inspect their text, or **e** to edit the proposed text directly.
- Accepted mappings are remembered across runs and methods, so the system improves its proposals over time.

If you work with something that does not have numbered segments, you will need to modify the script so it does not attempt to use the segment numbers. The mapping is ultimately of the text, not of segment numbers.

### Human evaluation

Human evaluation is done with the web tool in `web/`. Serve it with any static file server, e.g.:
```
python -m http.server 8000 --directory web
```
Then open `http://localhost:8000` in a browser. Results are downloaded as `session.json` and processed with `code/compute_human_eval_summary.py`.

### Evaluation

If you have reference translations, you can evaluate the translations obtained above automatically. You can also easily adapt the procedure to not require a reference by modifying the prompt. We did not test the quality/reliability of reference-free evaluation.

Generic form:
```
python code/evaluate_mqm_parallel.py <film_name> <source_lang> <target_lang> <trans_model> <eval_model> <eval_runs> <prompt_file>
```

For example:
```
python code/evaluate_mqm_parallel.py sample-ivan-vas Russian English gpt-5.2 gpt-5.4-mini 4 films/prompts/eval/mqm-memes.txt
```

This takes the challenging units and their translations from `films/output/translations/sample-ivan-vas/gpt-5.2.json` and evaluates them 4 independent times per translation using `gpt-5.4-mini` as the evaluator. Results are written to `films/output/eval/llm-eval/sample-ivan-vas/Russian-English/gpt-5.2-by-gpt-5.4-mini/`.

The evaluation is based on an adapted MQM (Multidimensional Quality Metric; see prompt). It is a flexible metric which you can modify for your needs.

You can restrict evaluation to specific methods or runs:
```
python code/evaluate_mqm_parallel.py sample-ivan-vas Russian English gpt-5.2 gpt-5.4-mini 4 films/prompts/eval/mqm-memes.txt --methods zero,given --runs 1,2,3
```

### Assessing evaluation reliability

You can assess how much you can trust the evaluation results. If the evaluation script ran successfully to the end, it will produce a summary file with statistics per method, including a confidence interval.

If the script crashed (for example, because the model returned something unexpected), you can compute statistics for the evaluation runs that did finish (and then add more runs if needed):

Generic form:
```
python code/aggregate_mqm.py <film_name> <trans_model>-by-<eval_model> <source_lang> <target_lang>
```

For example:
```
python code/aggregate_mqm.py sample-ivan-vas gpt-5.2-by-gpt-5.4-mini Russian English
```

This produces `merged_summary.json` and `method_comparison.json` in the eval directory. You can inspect these files for per-method statistics.

To determine whether you have enough translation and evaluation runs to reliably distinguish between methods, use the variance script:

Generic form:
```
python code/variance.py <film_name> <trans_model>-by-<eval_model> <delta> <source_lang> <target_lang>
```

For example:
```
python code/variance.py sample-ivan-vas gpt-5.2-by-gpt-5.4-mini 0.05 Russian English
```

`delta` is the smallest score difference you care about detecting. The script will tell you whether the current number of runs is sufficient, and whether you should add more translation runs or more evaluation runs first.

### Human evaluation web tool

The `web/` directory is a git submodule pointing to the `mqm-memes` repository. It hosts a static GitHub Pages site for human MQM evaluation of subtitle translations.

**After making changes in `web/`:**

```bash
# 1. Commit and push from inside the submodule
cd web
git add <files>
git commit -m "..."
git push

# 2. Update the submodule pointer in the parent repo
cd ..
git add web
git commit -m "Update web submodule: ..."
git push

# 3. If you also maintain a separate checkout of mqm-memes, pull there too
cd /path/to/mqm-memes
git pull
```

Step 2 is what records the new submodule commit in the parent repo so that anyone cloning `armen-subtitulos` gets the correct version of `web/`.

**To pull changes made elsewhere into `web/`:**

```bash
# Update the submodule to the latest remote commit
git submodule update --remote web

# Then record the updated pointer in the parent repo
git add web
git commit -m "Update web submodule"
git push
```
