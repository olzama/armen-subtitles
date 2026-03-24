## AI assisted creative translation and translation evaluation
The project consists of code which translates movie subtitles and evaluates the translations. It can easily be adapted to other domains, but is meant specifically for creative, highly challenging domains which do not yet allow fully automatic translation.

The code makes calls to AI APIs such as OpenAI and Gemini. You can easily adapt it to any AI API to which you have access.

```
pip install requirements.txt
```

### Translating
You can translate any text without any additional information provided to the translator:

```
python code/translate.py input_dir output_dir temperature num_translation_runs model_name 
```

For example:

```
python code/translate.py data/sample/ivan-vas/subs/ output/translations/zero/ 0.8 6 gpt-5.2
```

The above will produce 6 translations of the input text provided under data/sample/ using gpt-5.2 model (assuming you have an API key) using temperature=0.8. The prompt will be the basic prompt (found in the code) that basically just asks the model to translate from Russian into English.

Then you may use three other translation modes: 1) an additional summary mode; 2) a challenging unit list mode; and 3) a given translation mode.

Method 3 is a good way to integrate preapproved translations of challenging units into larger texts which still need to be translated. You need to have translated the units beforehand.
```
python code/translate.py data/sample/ivan-vas/subs/ output/translations/given/ 0.8 3 gpt-5.2 --prompt prompts/given_translations.txt --units_trans data/sample/ivan-vas/reference.json
```

Method 2 requires you to have identified the units which you think require special attention or are likely to not be translated well off the bat.
```
python code/translate.py data/sample/ivan-vas/subs/ output/translations/list/ 0.8 3 gpt-5.2 --prompt prompts/unit-list.txt --units_list data/sample/ivan-vas/meme_list.json
```

Method 1 is for experimentation with various prompts if you want to see whether or not a particular prompt improves translation quality.
```
python code/translate.py data/sample/ivan-vas/subs/ output/translations/characters/ 0.8 3 gpt-5.2 --prompt prompts/characters.txt --summary data/sample/ivan-vas/summaries/characters.txt
```

The above will save the required number of translations in the indicated output directories.

### Compiling data for evaluation
In order to save on API calls, we suggest you only evaluate the challenging units. The rest is very likely to be translated well anyway.

To evaluate economically against a reference, you need to first extract the translation portions corresponding to the challenging units. If you want to evaluate without a reference, we suggest you evaluate translations in full to get better results.

If you work with SRT subtitles with numbered segments and have a reference where those segments are indicated, you can use the script to map the translations to the original:
```
python code/map_translation_segments.py ivan-vas output/translations/ data/sample/ivan-vas/reference.json output/translations/ivan-vas-mapped.json gpt-5.2
```
This is an interactive script that will ask you to confirm the mapping or reject it and inspect surrounding segments to see if the correct segment is found nearby.

If you work with something that does not have numbered segments, you will need to map the translations of challenging segments to the originals yourself, if you want to save on API calls.

### Evaluation

If you have reference translations, you can evaluate the translations obtained as above automatically. You can also easily adapt the procedure to not require a reference by simply modifying the prompt. We did not test the quality/reliability of such evaluation.
```
python code/evaluate_mqm_parallel.py ivan-vas translations/gpt-5.2.json output/films/pokrov-gate/eval/ prompts/evaluation/mqm-memes.txt gpt-5-mini 4
```

The above will take the challenging units and their translations from the file translations/gpt-5.2.json and will evaluate them according to the prompt using gpt-5-mini as evaluator. It will perform the evaluation 4 independent times for each translation of each challenging unit.

The evaluation is based on an adapted MQM (Multidimensional Quality Metric; see prompt). It is a flexible metric which you can modify for your needs.

### Assessing evaluation reliability

You can assess how much you can trust the evaluation results. If the evaluation script ran successfully till the end, it will produce a summary file with the statistics per method, which will include a confidence interval. 

If the script crashed (for example, because the model returned something unexpected), you can obtain the statistics for the evaluation runs that did finish (and then add more runs if needed):
```
python code/aggregate_mqm.py output/translations/eval/ ivan-vas
```
This will produced a file named `merged_summary.json` and `method_comparison.json` in the eval directory. You can inspect these files for stats.

You can see whether you can compare different results to each other by assessing the variance of the evaluation scores:
```
python code/variance.py output/translations/eval/merged_summary.json 0.05
```
You will see if the number of translation and evaluation runs is enough for you to compare methods which yield scores that differ by at least 0.05.
