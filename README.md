## AI assisted creative translation and translation evaluation
The project consists of code which translates movie subtitles and evaluates the translations. It can easily be adapted to other domains, but is meant specifically for creative, highly challenging domains which do not yet allow fully automatic translation.

The code makes calls to AI APIs such as OpenAI and Gemini. You can easily adapt it to any AI API to which you have access.

```
pip install -r requirements.txt
```

### Directory structure

Input subtitles go in `data/films/<film_name>/subs/`. Outputs are written under `output/films/<film_name>/`.

```
data/films/<film_name>/subs/          ← source SRT file(s)
output/films/<film_name>/
    translations/<trans_model>/<method>/   ← raw translation runs (SRT/TXT)
    translations/<trans_model>.json        ← mapped translations (for evaluation)
    eval/<trans_model>-by-<eval_model>/    ← evaluation results
```

### Translating

Generic form:
```
python code/translate.py <film_name> <method> <trans_model> <temperature> <n_runs> [--prompt FILE] [--summary FILE] [--unit_list FILE] [--given_trans FILE]
```

`method` is a free label for the translation approach (e.g. `zero`, `summary`, `given`). It has no effect on the translation itself — it names the output subdirectory, which is then used in evaluation.

**Zero-shot (no additional context):**
```
python code/translate.py sample-ivan-vas zero gpt-5.2 0.8 6
```
Produces 6 translations under `output/films/sample-ivan-vas/translations/gpt-5.2/zero/` using the basic prompt (translate from Russian to English, return translation only).

**With a summary or character sheet:**
Useful for experimenting with prompts that provide background context.
```
python code/translate.py sample-ivan-vas characters gpt-5.2 0.8 3 --prompt prompts/characters.txt --summary data/films/sample-ivan-vas/summaries/characters.txt
```

**With a list of challenging units:**
Asks the model to pay special attention to culturally or linguistically difficult items identified in advance.
```
python code/translate.py sample-ivan-vas list gpt-5.2 0.8 3 --prompt prompts/unit-list.txt --unit_list output/films/sample-ivan-vas/meme-list.json
```

**With pre-approved translations of challenging units:**
Integrates previously approved translations of difficult items into the full translation. A good way to enforce consistency on items that need human oversight.
```
python code/translate.py sample-ivan-vas given gpt-5.2 0.8 3 --prompt prompts/given-translations.txt --given_trans output/films/sample-ivan-vas/reference.json
```

### Compiling data for evaluation

In order to save on API calls, we suggest you only evaluate the challenging units. The rest is very likely to be translated well anyway.

To evaluate economically against a reference, you need to first extract the translation portions corresponding to the challenging units. If you want to evaluate without a reference, we suggest you evaluate translations in full to get better results.

If you work with SRT subtitles with numbered segments and have a reference JSON where those segment numbers are indicated, you can use the interactive mapping script to align the translations to the original items:
```
python code/map_translation_segments.py <film_name> <trans_model>
```

For example:
```
python code/map_translation_segments.py sample-ivan-vas gpt-5.2
```

This reads the translated SRT files from `output/films/sample-ivan-vas/translations/gpt-5.2/` and writes the mapped translations to `output/films/sample-ivan-vas/translations/gpt-5.2.json`.

The script is interactive. For each item it proposes a segment mapping by searching for the translation segment most semantically similar to the reference translation (or to a previously accepted mapping for that item, if one exists). You review items in batches:
- Press **ENTER** to approve all items in a batch, or enter numbers (e.g. `2 5`) to flag specific items for correction.
- For a flagged item: press **ENTER** to accept the current proposal, **n** to widen the search window and get a new proposal, type segment number(s) to pick different segments and inspect their text, or **e** to edit the proposed text directly.
- Accepted mappings are remembered across runs and methods, so the system improves its proposals over time.

If you work with something that does not have numbered segments, you will need to modify the script so it does not attempt to use the segment numbers. The mapping is ultimately of the text, not of segment numbers.

### Evaluation

If you have reference translations, you can evaluate the translations obtained above automatically. You can also easily adapt the procedure to not require a reference by modifying the prompt. We did not test the quality/reliability of reference-free evaluation.

Generic form:
```
python code/evaluate_mqm_parallel.py <film_name> <trans_model> <eval_model> <eval_runs> <prompt_file>
```

For example:
```
python code/evaluate_mqm_parallel.py sample-ivan-vas gpt-5.2 gpt-4o-mini 4 prompts/evaluation/mqm-memes.txt
```

This takes the challenging units and their translations from `output/films/sample-ivan-vas/translations/gpt-5.2.json` and evaluates them 4 independent times per translation using `gpt-4o-mini` as the evaluator. Results are written to `output/films/sample-ivan-vas/eval/gpt-5.2-by-gpt-4o-mini/`.

The evaluation is based on an adapted MQM (Multidimensional Quality Metric; see prompt). It is a flexible metric which you can modify for your needs.

You can restrict evaluation to specific methods or runs:
```
python code/evaluate_mqm_parallel.py sample-ivan-vas gpt-5.2 gpt-4o-mini 4 prompts/evaluation/mqm-memes.txt --methods zero,given --runs 1,2,3
```

### Assessing evaluation reliability

You can assess how much you can trust the evaluation results. If the evaluation script ran successfully to the end, it will produce a summary file with statistics per method, including a confidence interval.

If the script crashed (for example, because the model returned something unexpected), you can compute statistics for the evaluation runs that did finish (and then add more runs if needed):

Generic form:
```
python code/aggregate_mqm.py <film_name> <trans_model>-by-<eval_model>
```

For example:
```
python code/aggregate_mqm.py sample-ivan-vas gpt-5.2-by-gpt-4o-mini
```

This produces `merged_summary.json` and `method_comparison.json` in the eval directory. You can inspect these files for per-method statistics.

To determine whether you have enough translation and evaluation runs to reliably distinguish between methods, use the variance script:

Generic form:
```
python code/variance.py <film_name> <trans_model>-by-<eval_model> <delta>
```

For example:
```
python code/variance.py sample-ivan-vas gpt-5.2-by-gpt-4o-mini 0.05
```

`delta` is the smallest score difference you care about detecting. The script will tell you whether the current number of runs is sufficient, and whether you should add more translation runs or more evaluation runs first.
