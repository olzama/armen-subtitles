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

## gentelmen, meme 6: "No movie, the electricity's run out"

Source: Все! Кина не будет. Электричество кончилось. (Nonstandard «кина»; electricity "used up" like a household supply; became a stock comic announcement that the show is over.)
Reference: That's it! No movie. The electricity's all gone.

**Highlighted (GPT-6 Sol, list, run 1):**
That's all, folks! The power's run out.

Note: "That's all, folks!" is the Looney Tunes sign-off (Warner Bros., Porky Pig), not Disney. Check wording in the paper.
It replaces a Russian catchphrase with an English-language one that does the same job (the show-is-over announcement), a target-culture substitution.

Same meme, other translators:
- GPT-5.2: always the literal "That's it! No movie. ..." in every condition and run.
- GPT-6 Sol, zero: "That's it! Show's over. ..." (idiomatic, no catchphrase).
- GPT-6 Sol, list: "That's all, folks!" in all 3 runs (also characters run 3). list-analysis: run 1 uses it; runs 2-3 go for the nonstandard register instead ("No more movin' pictures"), which mirrors «кина».
- Opus 5.5: literal in zero and list; list-analysis renders the register ("No more picture show. Electricity's all used up.").

Pattern: being told it is a meme triggers a target-culture catchphrase substitution; the analysis (which stresses the nonstandard «кина») pulls some runs toward reproducing the register instead.
