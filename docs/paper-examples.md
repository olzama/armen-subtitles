# Translation examples to cite in the paper

Collected while doing the human evaluation of GPT-6 Sol and Opus 5.5 (from 2026-10-19). Olga: "with earlier models, we were not getting anything like this. We will mention this in the paper. It is not the only great example."

## gentelmen, meme 1: the Big Bad Wolf rhyme

Source: Я злой и страшный серый волк, я в поросятах знаю толк! (Troshkin performing the wolf at the kindergarten; rhymed children's-play menace.)
Reference: I'm the Big Bad Wolf, and I know a piglet or two!

**Highlighted (Opus 5.5, list-analysis, run 1):**
I'm the big bad wolf, so grim and gruff, / and piglets? Oh, I know my stuff!

Same meme, other translators:
- GPT-5.2: no rhyme in any condition/run, e.g. "I'm the mean and scary Big Bad Wolf, / I know my way around piglets." (list-analysis 1); zero 2 even leaves "злой" untranslated.
- GPT-6 Sol, zero: no rhyme ("I'm the big, bad, fearsome wolf. / I know a tasty pig when I see one.").
- GPT-6 Sol, list and list-analysis: rhymes in most runs, e.g. "I'm a fearsome wolf, all teeth and snout; / I know what little pigs are all about!" (list-analysis 2).

Pattern worth stating: current models recover the form (rhyme) once told it is a meme; GPT-5.2 never does, even with the analysis.
