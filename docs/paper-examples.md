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
- Opus 5.5: rhymes in zero ("I'm the wolf so big and bad, / the tastiest piglets I've ever had."), list and list-analysis; not in characters.
- GPT-6 Sol, list and list-analysis: rhymes in most runs, e.g. "I'm a fearsome wolf, all teeth and snout; / I know what little pigs are all about!" (list-analysis 2).

Pattern worth stating: GPT-5.2 never recovers the rhyme, even with the analysis; GPT-6 Sol does once told it is a meme; Opus 5.5 rhymes it even unaided (except with the character list).

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

## gentelmen, meme 8: the monument dialogue ("Сидит? ... Кто же его посадит?")

Source: — Пруд там был? — Не было. Лужи были. Может памятник? ... — Сидячий? — Чего? — Сидит? — Кто? — Мужик этот твой! — Во деревня! Ну ты даешь! Кто же его посадит? Он же памятник!
Reference: Was there a square? — There was. — Maybe a monument? ... — Was he inside? — Who? — That guy of yours! — What a hick! Listen to you! Who'd put him inside? He's a monument!
The analysis says English cannot reproduce the сидеть/посадить root play and that the approved version "builds a compensating ambiguity around being 'inside'".

**Highlighted (GPT-6 Sol, list-analysis, run 1; all 3 runs do it):**
- Was there a pond? / - No. There were puddles. / A monument, maybe? / - There was a monument. / - To whom? / - How should I know? Some man. / - Beard? - No! / - Sideburns? / - I don't remember. He wore a jacket! / - Was he inside? - What? / - Inside? - Who? / - That man of yours! / - What a country bumpkin! Honestly! / Who's gonna put him inside? / He's a monument!

Olga: fantastic even though it differs from the reference. It differs in surface choices (keeps the literal pond/puddles where the reference has a square; "country bumpkin"), but it lands the pun with the compensation the analysis names.

Same meme, other translators:
- GPT-5.2: "Sitting?/Seated?" in every condition and run, including list-analysis: the pun is lost even though the analysis names the fix.
- GPT-6 Sol, zero and list: "Seated?/Sitting down?" (pun lost).
- GPT-6 Sol, list-analysis: "Was he inside?" in 3/3 runs.
- Opus 5.5: "Seated?/Sitting down?" in all conditions, including list-analysis.

Pattern: cleanest single illustration of the "using the analysis" finding. The same analysis is given to every model; only the newest GPT model turns it into the pun, consistently, and keeps its own wording elsewhere instead of copying the reference.

## diamond-arm, meme 7: Kutuzov → Nelson (flagged by Olga during the Opus meme-search scoring, 2026-10-20: "that is fantastic!!")

Source: Дитя́м — мороженое, его бабе — цветы. Смотры, нэ пэрэпутай!.. Кутузоу! (Lyolik to the bandaged, one-eyed Gesha; non-standard speech; "Kutuzov" mocks the black eye via the one-eyed Napoleonic-era general.)
Reference: Ice cream for the young'uns, flowers for his broad. Look now, don't mix 'em up! Kutuzov!

**Highlighted (Opus 5.5, meme-search, all three runs):** ice cream for the kiddies, / flowers for his missus. / Don't mix it up, / Nelson! (run 1)

- Where it comes from: Opus's own meme-search guide (meme-analysis.txt, segments 528–535): "A one-eyed general, because Gesha has a black eye. English viewers won't recognise Kutuzov." → "…Don't mix it up, Nelson!" or keep "Kutuzov". The translations follow the guide in 3/3 runs.
- Every other condition and translator keeps "Kutuzov" (Opus zero/characters/list/list-analysis/given; GPT-6 Sol all conditions incl. meme-search; GPT-5.2 all conditions). The approved analysis names Kutuzov and does not suggest an English substitute.
- Same guide, same move: it also turns the Kolyma camps into "Siberia" ("Будете у нас на Колыме…"), which Olga judges better understood by an English viewer (2026-10-20), while every automatic evaluator penalizes it. The paper had framed this as a misfire; corrected.
- **Evaluators:** Sonnet 5.5 (main) marks "Nelson" a *major accuracy* error in 6/6 evaluations, even while calling it "a reasonable equivalent" / "a creative equivalent for an English audience", because the analysis names Kutuzov. GPT-5-mini: mostly minor style (4/6 minor, 2/6 major). Haiku: minor at most, often only for the lost non-standard speech.
- **For the paper:** the item-level evaluator penalizes a good creative substitution that departs from the analysis/reference — the reference-adherence failure documented for the whole-text judge ("Оставь меня, старушка"), now on Meme-MQM. Compare with Olga's human score once the session is in.

## diamond-arm, meme 8: the teetotaler toast, rhymed at last (flagged by Olga, 2026-10-20: "finally making it sound great")

Source: Как говорит наш дорогой шеф, за чужой счёт пьют даже трезвенники… и язвенники! (rhyme and matching suffixes: трезвенники / язвенники)
Reference: On someone else's dime, even teetotalers and convalescers drink! (no rhyme; the approved analysis's English note: "no rythm whatsover; at least some creative solution must be present.")

**Highlighted (Opus 5.5, meme-search, all three runs):** As our dear chief says, / on someone else's dime, the teetotaller / and the ulcer case both have a fine time.

- From Opus's own meme-search guide (segments 738–740): "The rhyme matters more than literal accuracy."
- Other Opus conditions try once each: "ascetics … diabetics" (list 2), "teetotalers … ulcer-totalers" (list-analysis 3); the rest are unrhymed. No GPT-6 Sol (18) or GPT-5.2 (56) rendering rhymes.
- **Evaluator:** Sonnet scores it 0 or one minor error ("a creative rhyme … which is welcome"; "answers the note about rhythm"). Contrast with Kutuzov → Nelson: the evaluator rewards a departure the analysis invites, and penalizes one where the analysis names the original reference.
