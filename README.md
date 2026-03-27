## AI assisted creative translation and translation evaluation
The project consists of code which translates movie subtitles and evaluates the translations. It can easily be adapted to other domains, but is meant specifically for creative, highly challenging domains which do not yet allow fully automatic translation.

The code makes calls to AI APIs such as OpenAI and Gemini. You can easily adapt it to any AI API to which you have access.

```
pip install requirements.txt
```

### Translating
You can translate any text without any additional information provided to the translator:

```
python code/translate.py film_name model_name method_name temperature num_translation_runs  
```

The "method_name" parameter does not have any effect; it is used to direct the output into the right directory. You can call the method whatever you like.

For example:

```
python code/translate.py sample-ivan-vas zero 0.8 6 gpt-5.2
```

The above will produce 6 translations of the input text provided under data/films/sample-ivan-vas/ using gpt-5.2 model (assuming you have an API key) using temperature=0.8. The prompt will be the basic prompt (found in the code of `translate.py`) that basically just asks the model to translate from Russian into English and to not stop midway and to not output anything but the translation.

Then you may use three other translation modes: 1) an additional summary mode; 2) a challenging unit list mode; and 3) a given translation mode.

Method 3 is a good way to integrate preapproved translations of challenging units into larger texts which still need to be translated in full. You need to have translated the challenging units beforehand.
```
python code/translate.py sample-ivan-vas given 0.8 3 gpt-5.2 --prompt prompts/given_translations.txt --units_trans data/sample-ivan-vas/reference.json
```

Method 2 requires you to have identified the units which you think require special attention or are likely to not be translated well off the bat.
```
python code/translate.py sample-ivan-vas list 0.8 3 gpt-5.2 --prompt prompts/unit-list.txt --units_list data/sample-ivan-vas/meme_list.json
```

Method 1 is for experimentation with various prompts if you want to see whether or not a particular prompt improves translation quality.
```
python code/translate.py sample-ivan-vas characters 0.8 3 gpt-5.2 --prompt prompts/characters.txt --summary data/sample-ivan-vas/summaries/characters.txt
```

The above will save the required number of translations in the output directories named after method. These directory names should then be used in evaluation.

### Compiling data for evaluation
In order to save on API calls, we suggest you only evaluate the challenging units. The rest is very likely to be translated well anyway.

To evaluate economically against a reference, you need to first extract the translation portions corresponding to the challenging units. If you want to evaluate without a reference, we suggest you evaluate translations in full to get better results.

If you work with SRT subtitles with numbered segments and have a reference where those segments are indicated, you can use the script to map the translations to the original:
```
python code/map_translation_segments.py sample-ivan-vas gpt-5.2
```
This is an interactive script that will ask you to confirm the mapping or reject it and inspect surrounding segments to see if the correct segment is found nearby. It will save output in a JSON file named after the model_name.

If you work with something that does not have numbered segments, you will need to modify the script so it does not attempt to use the segment numbers, but it should not be difficult. The mapping is ultimately of the text, not segments.

### Evaluation

If you have reference translations, you can evaluate the translations obtained as above automatically. You can also easily adapt the procedure to not require a reference by simply modifying the prompt. We did not test the quality/reliability of such evaluation.
```
python code/evaluate_mqm_parallel.py sample-ivan-vas gpt-5.2.json gpt-5-mini prompts/evaluation/mqm-memes.txt 4
```

The above will take the challenging units and their translations from the file translations/gpt-5.2.json and will evaluate them according to the prompt using gpt-5-mini as evaluator. It will perform the evaluation 4 independent times for each translation of each challenging unit.

The evaluation is based on an adapted MQM (Multidimensional Quality Metric; see prompt). It is a flexible metric which you can modify for your needs.

### Assessing evaluation reliability

You can assess how much you can trust the evaluation results. If the evaluation script ran successfully till the end, it will produce a summary file with the statistics per method, which will include a confidence interval. 

If the script crashed (for example, because the model returned something unexpected), you can obtain the statistics for the evaluation runs that did finish (and then add more runs if needed):
```
python code/aggregate_mqm.py sample-ivan-vas gpt-5-mini
```
This will produced a file named `merged_summary.json` and `method_comparison.json` in the `eval` directory. You can inspect these files for stats.

You can see whether you can compare different results to each other by assessing the variance of the evaluation scores:
```
python code/variance.py sample-ivan-vas gpt-5-mini 0.05
```
You will see if the number of translation and evaluation runs is enough for you to compare methods which yield scores that differ by at least 0.05.
