const slides = [
  {
    title: "The Pipeline at a Glance",

    content: () => `
      <p class="caption">
        The pipeline translates a set of culturally challenging subtitle lines
        using several different methods, then evaluates and compares the results
        statistically. The goal is to find out which method produces the best
        translations — and to be confident enough in that answer to publish it.
      </p>

      <div class="flow">
        <div class="flow-step">
          <div class="flow-box">.srt file</div>
          <div class="flow-label">source subtitles</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
          <div class="flow-box">challenging<br>units</div>
          <div class="flow-label">reference.json<br>(human curated)</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
          <div class="flow-box highlight">translate</div>
          <div class="flow-label">T runs × M methods</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
          <div class="flow-box highlight">map</div>
          <div class="flow-label">align to units<br>(interactive)</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
          <div class="flow-box highlight">evaluate</div>
          <div class="flow-label">E runs × T × M<br>MQM scoring</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
          <div class="flow-box highlight">variance</div>
          <div class="flow-label">enough data?<br>update YAML</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
          <div class="flow-box">results</div>
          <div class="flow-label">method comparison<br>with CI</div>
        </div>
      </div>

      <p class="caption">
        The dark boxes are automated (run by the pipeline driver).
        The <em>map</em> step is interactive — a human reviews the segment
        alignments before evaluation begins.
        Everything is driven by a single YAML config file.
      </p>
    `
  },

  {
    title: "Two Phases",
    content: () => `
      <p class="caption">
        The work divides into two phases. The first is done once, by hand, before
        the pipeline runs. The second is automated and iterative.
      </p>

      <div class="flow" style="flex-direction:column; gap:1rem;">
        <div style="border:2px solid #1a1a1a; border-radius:6px; padding:1rem 1.25rem;">
          <div style="font-family:'Courier New',monospace; font-size:0.85rem; font-weight:bold; margin-bottom:0.5rem;">
            PHASE 1 — Building the dataset &nbsp;<span style="font-weight:normal; color:#888;">(2–4 hours per film, done once)</span>
          </div>
          <ol style="margin-left:1.2rem; font-size:0.92rem; line-height:2;">
            <li>Pick a film in English known for memorable quotes and cultural memes</li>
            <li>Collect a list of particularly popular quotes from the film (e.g. from Wikiquote)</li>
            <li>Guide a good LLM (GPT or Gemini, chat version) to provide a cultural analysis
                of each quote: why is it memorable, what emotions does it provoke,
                in what situations do people quote it</li>
            <li>Guide the LLM to produce reference translations for each item in Galician
                and/or Spanish — a translation that <em>works</em>: is funny, could become
                memorable, relates to something culturally meaningful for the target audience</li>
          </ol>
        </div>

        <div style="border:2px solid #1a1a1a; border-radius:6px; padding:1rem 1.25rem; background:#1a1a1a; color:#f5f4f0;">
          <div style="font-family:'Courier New',monospace; font-size:0.85rem; font-weight:bold; margin-bottom:0.5rem;">
            PHASE 2 — Running the experiment &nbsp;<span style="font-weight:normal; color:#aaa;">(automated, repeatable)</span>
          </div>
          <ol style="margin-left:1.2rem; font-size:0.92rem; line-height:2;">
            <li>The full subtitles are translated multiple times using several different methods</li>
            <li>The challenging segments are automatically extracted from each translation
                and mapped to the reference units — you review and approve the mappings,
                as there may be errors (done once)</li>
            <li>The approved segments are scored by a separate LLM judge</li>
            <li>The pipeline checks whether the scores are stable enough to trust,
                and tells you if you need to run more translations or more evaluations</li>
            <li>The result is a comparison of methods with confidence measures</li>
          </ol>
        </div>
      </div>

      <p class="caption">
        The following slides walk through each phase step by step,
        using a real example from <em>Pulp Fiction</em>.
      </p>
    `
  },

  {
    title: "Challenging Units",
    content: () => `
      <p class="caption">
        Not every subtitle line is hard to translate. The pipeline focuses on
        <em>challenging units</em> — lines that involve cultural references,
        wordplay, or register that a generic model is likely to handle poorly.
        These are curated by hand and stored in <code>reference.json</code>
        along with reference translations and an explanation of the challenge.
      </p>

      <p class="caption note">
        <strong>This example was generated entirely by an LLM (Claude) and has not
        been reviewed — it likely contains errors, especially in Galician.</strong>
        In practice, building <code>reference.json</code> is a guided,
        human-in-the-loop process that takes roughly 2–4 hours per film:
        (1) collect memorable quotes from Wikiquote;
        (2) work through the quotes one by one with the LLM — it proposes the
        cultural analysis for each item, you read it, push back, correct, and
        approve (see next slides for examples);
        (3) the LLM then proposes reference translations, again one by one, and
        you steer and approve each one.
        At every stage the LLM is doing the drafting and you are making the calls.
      </p>

      <pre class="code-block">{
  <span class="key">"id"</span>: <span class="num">1</span>,
  <span class="key">"character"</span>: <span class="str">"Vincent"</span>,

  <span class="key">"original"</span>: {                           <span class="cmt">← source text, keyed by language code</span>
    <span class="key">"eng"</span>: <span class="str">"And you know what they call a Quarter
           Pounder with cheese in Paris? Got the
           metric system there. They wouldn't know
           what the fuck a Quarter Pounder is."</span>
  },

  <span class="key">"reference"</span>: {                          <span class="cmt">← human-approved translations</span>
    <span class="key">"spa"</span>: <span class="str">"¿Sabes cómo llaman al Cuarto de Libra
           con Queso en París? Allí tienen el sistema
           métrico; no sabrían lo que es un cuarto
           de libra."</span>,
    <span class="key">"glg"</span>: <span class="str">"¿Sabes como lle chaman ao Cuarto de Libra
           con Queso en París? Alí teñen o sistema
           métrico; non saberían o que é un cuarto
           de libra."</span>
  },

  <span class="key">"analysis"</span>: {                           <span class="cmt">← fed to the model as context</span>
    <span class="key">"general"</span>: {
      <span class="key">"text"</span>: <span class="str">"Vincent treats this as a profound
              philosophical discovery about cultural
              difference, not a mundane marketing fact.
              The translator must preserve his tone."</span>,
      <span class="key">"nb"</span>:   <span class="str">"The actual Spanish McDonald's name was
              'McRoyal con Queso'. A translator may
              choose to use this, or keep 'Cuarto de
              Libra' to preserve Vincent's logic."</span>
    }
  },

  <span class="key">"segment_number"</span>: [<span class="num">133</span>, <span class="num">134</span>, <span class="num">135</span>]         <span class="cmt">← SRT line numbers</span>
}</pre>

      <p class="caption">
        The <code>analysis</code> field is the key differentiator between
        translation methods: a zero-shot model never sees it, while the
        <em>list-analysis</em> method receives it as part of the prompt.
      </p>
    `
  },

  {
    title: "Building the Analysis — Human in the Loop",
    content: () => `
      <p class="caption">
        For each quote, you ask the LLM to explain why it is memorable —
        what makes it funny, resonant, or culturally loaded.
        The LLM proposes; you read carefully, push back where it is wrong or
        shallow, and approve when it is right. Below is an example exchange
        for <em>"Zed's dead, baby. Zed's dead."</em>
        Expect roughly <strong>5–10 minutes per item</strong>, or
        <strong>1–2 hours</strong> for a list of ~10 quotes.
      </p>

      <div class="chat">
        <div class="chat-turn human">
          <div class="chat-label">You</div>
          <div class="chat-bubble">
            Here is a quote from Pulp Fiction (1994):<br><br>
            <strong>Butch:</strong> "Zed's dead, baby. Zed's dead."<br><br>
            Why did this line become a meme? What makes it memorable?
          </div>
        </div>

        <div class="chat-turn llm">
          <div class="chat-label">LLM</div>
          <div class="chat-bubble">
            The line works because of the gap between what is being said and how
            it is said. Butch has just killed a man, yet his delivery is completely
            calm — almost gentle. The repetition ("Zed's dead... Zed's dead")
            gives it a quiet, rhythmic finality, like closing a door.
            "Baby", addressed to Fabienne, adds an unexpected tenderness.
            The result is a line that is simultaneously brutal and intimate,
            which is why it lodged in popular culture.
          </div>
        </div>

        <div class="chat-turn human">
          <div class="chat-label">You</div>
          <div class="chat-bubble">
            Good. Add something about the rhythm — it almost sounds like a lullaby.
          </div>
        </div>

        <div class="chat-turn llm">
          <div class="chat-label">LLM</div>
          <div class="chat-bubble">
            Yes — the stress pattern of "Zed's dead, baby" is identical to a
            lullaby cadence (DUM-dum DUM-dum). Tarantino almost certainly chose
            it deliberately. It turns a death announcement into something soothing,
            which deepens the dark comedy and explains why the line feels so
            complete and self-contained.
          </div>
        </div>

        <div class="chat-turn human">
          <div class="chat-label">You</div>
          <div class="chat-bubble">
            Approved. Write the final analysis entry.
          </div>
        </div>
      </div>
    `
  },

  {
    title: "Building the Reference Translations — Human in the Loop",
    content: () => `
      <p class="caption">
        Once the analysis is approved, you ask the LLM for a reference translation.
        This is the human-approved target you will later use to evaluate the
        pipeline's output. The same back-and-forth applies: the LLM drafts,
        you steer. This step typically adds another
        <strong>1–2 hours</strong> for ~10 items.
      </p>

      <div class="chat">
        <div class="chat-turn human">
          <div class="chat-label">You</div>
          <div class="chat-bubble">
            Now translate into Galician:<br><br>
            <strong>Butch:</strong> "Zed's dead, baby. Zed's dead."<br><br>
            Keep in mind the lullaby rhythm and the tenderness of "baby".
          </div>
        </div>

        <div class="chat-turn llm">
          <div class="chat-label">LLM</div>
          <div class="chat-bubble">
            <em>Zed está morto, cariño. Zed está morto.</em><br><br>
            "Cariño" preserves the warmth of "baby" and is short enough not to
            disrupt the rhythm. "Morto" is the correct Galician form.
          </div>
        </div>

        <div class="chat-turn human">
          <div class="chat-label">You</div>
          <div class="chat-bubble">
            "Cariño" feels too Castilian. We want something more native to Galician,
            and a bit more informal — closer to how Butch actually speaks.
          </div>
        </div>

        <div class="chat-turn llm">
          <div class="chat-label">LLM</div>
          <div class="chat-bubble">
            <em>Zed está morto, nena. Zed está morto.</em><br><br>
            "Nena" (lit. girl, but used as a gender-neutral endearment in colloquial
            Galician) is more informal and more distinctly Galician than "cariño".
            It also fits the stress pattern of the original better —
            one short syllable, same position in the line.
          </div>
        </div>

        <div class="chat-turn human">
          <div class="chat-label">You</div>
          <div class="chat-bubble">
            Perfect. Approved.
          </div>
        </div>
      </div>
    `
  },

  {
    title: "Configuring and Running the Pipeline",
    content: () => `
      <p class="caption">
        Before writing the YAML, extract the analysis from your
        <code>reference.json</code> into a separate file that the
        <em>list-analysis</em> method will receive as context:
      </p>

      <pre class="code-block"><span class="dim">$</span> python code/select_from_json.py experiments/films/data/pulp-fiction/reference.json
<span class="cmt">→ experiments/films/data/pulp-fiction/list-analysis-general-all.json</span></pre>

      <p class="caption">
        This strips out the reference translations and keeps only the source
        text and the cultural analysis — exactly what the model needs to do a
        better-informed translation without being given the answer.
        Now define the experiment in a YAML file. It specifies the
        film, the language pair, the model, and the translation methods to compare.
        The pipeline driver reads it at every step, so the config is the single
        source of truth for what has been run and what still needs to be done.
      </p>

      <pre class="code-block"><span class="cmt"># yaml-pipelines/films/pulp-fiction-eng-glg.yaml</span>

<span class="key">film</span>:            <span class="str">pulp-fiction</span>
<span class="key">source_lang</span>:     <span class="str">English</span>
<span class="key">target_lang</span>:     <span class="str">Galician</span>
<span class="key">trans_model</span>:     <span class="str">gpt-5.2</span>
<span class="key">eval_model</span>:      <span class="str">gpt-5.4-mini</span>
<span class="key">variance_delta</span>:  <span class="num">0.1</span>      <span class="cmt">← target sensitivity (see last slide)</span>

<span class="key">methods</span>:
  - <span class="key">name</span>: <span class="str">zero</span>          <span class="cmt">← no context given to the model</span>
    <span class="key">n_runs</span>: <span class="num">3</span>
  - <span class="key">name</span>: <span class="str">list-analysis</span>  <span class="cmt">← cultural analysis provided</span>
    <span class="key">n_runs</span>: <span class="num">3</span>
    <span class="key">unit_list</span>: <span class="str">experiments/films/data/pulp-fiction/list-analysis-general-all.json</span></pre>

      <p class="caption note">
        The pipeline makes calls to the OpenAI API for both translation and evaluation.
        You will need an API key — we have one you can use for this project.
      </p>

      <p class="caption">
        Once the config is in place, the entire pipeline runs from a single command:
      </p>

      <pre class="code-block"><span class="dim">$</span> python run_pipeline.py yaml-pipelines/films/pulp-fiction-eng-glg.yaml</pre>

      <p class="caption">
        This translates the full subtitles with each method, prompts you to review
        the segment mappings, runs the LLM evaluation, and reports whether the
        results are reliable. Individual steps can also be run separately
        (e.g. <code>--step translate</code>, <code>--step eval</code>).
      </p>
    `
  },

  {
    title: "The Mapping Step",
    content: () => `
      <p class="caption">
        After translating, the pipeline needs to find, within each full subtitle
        file, the segments that correspond to your challenging units.
        It does this automatically using semantic similarity — but the match
        is not always perfect, so you review each proposal and approve or correct it.
        This is done once per translation run and should not take more than a few minutes.
      </p>

      <pre class="code-block"><span class="dim">$</span> python code/map_translation_segments.py pulp-fiction gpt-5.2 English Galician

================================================================================
METHOD: zero | RUN: 1 | Batch 1/2
================================================================================

  [1] Vincent (seg 133–135)
       Segments: [133, 134, 135]
       REF:      ¿Sabes como lle chaman ao Cuarto de Libra con Queso en París?
                 Alí teñen o sistema métrico; non saberían o que é un cuarto de libra.
       MAP:      E sabes como lle chaman a iso en París? Teñen o sistema métrico,
                 non saberían o que é un cuarto de libra.

  [2] Butch (seg 1585–1586)
       Segments: [1585, 1586]
       REF:      Zed está morto, nena. Zed está morto.
       MAP:      Zed está morto, nena. Zed está morto.

Press ENTER to approve all, b=back, or enter numbers to flag (e.g. 2 5): <span class="hl">ENTER</span>

================================================================================
METHOD: zero | RUN: 1 | Batch 2/2
================================================================================

  [1] Jules (seg 364)
       Segments: [363]  <span class="cmt">← off by one</span>
       REF:      ¡Inglés, fillo de puta! ¿Fálalo?
       MAP:      Que?

  [2] Wolf (seg 1822)
       Segments: [1822]
       REF:      Son Winston Wolf. Resolvo problemas.
       MAP:      Son Winston Wolf. Resolvo problemas.

Press ENTER to approve all, b=back, or enter numbers to flag (e.g. 2 5): <span class="hl">1</span>

================================================================================
METHOD: zero | RUN: 1 | ITEM: Jules (seg 364)
--------------------------------------------------------------------------------
REFERENCE TRANSLATION:
¡Inglés, fillo de puta! ¿Fálalo?
--------------------------------------------------------------------------------
NEARBY SEGMENTS:
   [361]  O que?
   [362]  O que?
<span class="hl">>>></span> [363]  Que?
   [364]  - ¡Inglés, fillo de puta! ¿Fálalo? - ¡Si!   <span class="cmt">← clearly the right one</span>
   [365]  - Entón xa sabes o que estou dicindo.
--------------------------------------------------------------------------------
PROPOSED MAPPED TEXT (segments [363]):
Que?
--------------------------------------------------------------------------------
ENTER=accept  b=back  n=widen search  e=edit/trim  [segment numbers]=pick
<span class="dim">> </span><span class="hl">364</span>
  MAP:  - ¡Inglés, fillo de puta! ¿Fálalo? - ¡Si!   <span class="cmt">← right segment, but extra text</span>
<span class="dim">> </span><span class="hl">e</span>
  <span class="cmt">  editor opens — trim to: ¡Inglés, fillo de puta! ¿Fálalo?</span>
  <span class="cmt">✓ accepted</span></pre>

      <p class="caption">
        REF is the reference translation you approved in Phase 1 — it tells you
        which source line you are looking for.
        MAP is the segment the pipeline found in the translated subtitle file.
        Your only job here is to confirm the right line was found:
        if MAP covers the same source dialogue as REF, approve it.
        Do not evaluate the translation quality — that is the evaluator's job.
        If the wrong line was found, type the correct segment number to replace it.
        If the right line is there but includes extra surrounding dialogue,
        use <code>e</code> to trim it.
      </p>
    `
  },

  {
    title: "Evaluating the Translations",
    content: () => `
      <p class="caption">
        After the mapping step, the pipeline asks a second LLM — the <em>judge</em> —
        to score each candidate translation.
        The judge receives the source text, the approved reference translation,
        the cultural analysis, and the candidate.
        It outputs a structured list of errors, categorised by severity.
        This is called <em>MQM evaluation</em> (Multidimensional Quality Metrics).
      </p>

      <pre class="code-block"><span class="cmt">← What the judge receives (one item):</span>

{
  <span class="key">"source"</span>:    <span class="str">"And you know what they call a Quarter Pounder
              with cheese in Paris? Got the metric system there.
              They wouldn't know what the fuck a Quarter Pounder is."</span>,

  <span class="key">"reference"</span>: <span class="str">"¿Sabes como lle chaman ao Cuarto de Libra con Queso
              en París? Alí teñen o sistema métrico; non saberían
              o que é un cuarto de libra."</span>,

  <span class="key">"analysis"</span>:  <span class="str">"Vincent treats this as a profound philosophical
              discovery about cultural difference. The translator
              must preserve his mock-philosophical tone."</span>,

  <span class="key">"candidate"</span>: <span class="str">"¿Sabes lo que llaman a eso en París? Tienen el
              sistema métrico, no sabrían lo que es una
              hamburguesa grande."</span>
}</pre>

      <pre class="code-block"><span class="cmt">← Judge output:</span>

{
  <span class="key">"id"</span>: <span class="num">1</span>,
  <span class="key">"issues"</span>: [
    {
      <span class="key">"severity"</span>: <span class="str">"major"</span>,
      <span class="key">"category"</span>: <span class="str">"accuracy"</span>,
      <span class="key">"justification"</span>: <span class="str">"'Hamburguesa grande' loses the central joke:
          Vincent uses the specific brand name as evidence of cultural
          difference. The name must be preserved."</span>
    },
    {
      <span class="key">"severity"</span>: <span class="str">"minor"</span>,
      <span class="key">"category"</span>: <span class="str">"style"</span>,
      <span class="key">"justification"</span>: <span class="str">"Register is slightly flat — Vincent's
          mock-philosophical tone does not come through."</span>
    }
  ]
}</pre>

      <p class="caption">
        Issues are weighted: <strong>minor = 1 pt, major = 5 pts</strong>.
        The total is divided by the major weight and by the number of units,
        giving a score in <em>major-equivalents per unit</em> — lower is better.
        Here: (5 + 1) &divide; 5 &divide; 1 unit = <strong>1.2</strong>.
        A translation with no issues scores <strong>0.0</strong>.
        Each translation is evaluated E times (set by <code>eval_runs</code> in the YAML)
        to average out the judge's own variability across runs.
      </p>
    `
  },

  {
    title: "Results and Reliability",
    content: () => `
      <p class="caption">
        The pipeline computes a mean score and a 95% confidence interval (CI) for each method,
        pooling all T translation runs and all E evaluation runs.
        If the CI is too wide to trust the ranking, the pipeline tells you to add more runs
        and updates the YAML automatically.
        The target CI width is set by <code>variance_delta</code> in the config.
      </p>

      <pre class="code-block"><span class="cmt">← First pass (T=3, E=5): CIs too wide to rank the methods</span>

METHOD           MEAN     95% CI       T    E
──────────────────────────────────────────────────
zero             0.31   &plusmn; 0.09        3    5
list-analysis    0.12   &plusmn; 0.07        3    5

<span class="cmt">← CI exceeds target (0.05). Recommended: increase T.</span>
<span class="cmt">← YAML updated automatically. Re-run the pipeline.</span>

<span class="cmt">← After adding runs (T=6, E=5):</span>

METHOD           MEAN     95% CI       T    E
──────────────────────────────────────────────────
zero             0.29   &plusmn; 0.04  <span class="hl">&#10003;</span>  6    5
list-analysis    0.11   &plusmn; 0.03  <span class="hl">&#10003;</span>  6    5

<span class="cmt">← CIs do not overlap → list-analysis significantly outperforms zero</span></pre>

      <p class="caption">
        A method wins when its CI does not overlap with another method's CI.
        The lower the score, the fewer translation errors the judge found.
        The <em>list-analysis</em> method — which receives the cultural analysis
        you prepared in Phase 1 — is expected to outperform the zero-shot baseline
        on the challenging units, because the model has the context it needs
        to make culturally informed choices.
      </p>

      <p class="caption note">
        The numbers above are illustrative. Real results depend on the film,
        the language pair, the model, and the quality of the reference data.
        The pipeline is designed to tell you when you have enough data to trust
        the result — so that the conclusion is defensible, not just plausible.
      </p>
    `
  },

  {
    title: "Where to Go From Here",
    content: () => `
      <p class="caption">
        The experiments described in this tutorial compare two methods:
        a <em>zero-shot</em> baseline (no context given to the model)
        and <em>list-analysis</em> (the model receives the cultural analysis
        you prepared in Phase 1).
        These are not the only options the pipeline supports.
      </p>

      <p class="caption">
        One natural direction for a new project is to experiment with
        <strong>N-shot prompting</strong>: instead of giving the model a literary
        analysis, you give it a small number of example translations drawn from
        an already existing dataset — in this case, the Russian film dataset.
        The datasets are very different culturally, so this is not about teaching
        the model the specific style of the film.
        It is an experiment in whether there are underlying cross-cultural
        similarities in the way memes work — and whether a model can pick up
        on those similarities from examples alone.
      </p>

      <p class="caption">
        The pipeline already supports adding new methods via the YAML config
        and a prompt file — no changes to the pipeline code are needed.
        The statistical machinery (multiple runs, MQM evaluation, variance check,
        confidence intervals) works the same way regardless of what the method does.
        Design clean prompts with varying N
        (idea: use the items from the Russian dataset that cause the least — or the most — variance as your examples),
        run the pipeline, analyze the result, ship it!
      </p>

      <p class="caption">
        The repository is at
        <strong>github.com/olzama/armen-subtitles</strong>.
        The top-level structure is:
      </p>

      <pre class="code-block">code/                        <span class="cmt">← pipeline scripts (translate, map, evaluate, variance)</span>
experiments/
  films/
    data/&lt;film&gt;/             <span class="cmt">← SRT files, reference.json, summaries</span>
    prompts/                 <span class="cmt">← translation and evaluation prompt files</span>
    output/                  <span class="cmt">← translations, mapped JSON, eval results</span>
    tutorial/                <span class="cmt">← this tutorial</span>
  armen/                     <span class="cmt">← YouTube-show dataset (separate pipeline)</span>
yaml-pipelines/
  films/&lt;film&gt;-&lt;src&gt;-&lt;tgt&gt;.yaml   <span class="cmt">← one config file per experiment</span>
run_pipeline.py              <span class="cmt">← pipeline driver (reads the YAML, runs all steps)</span>
README.md                    <span class="cmt">← full documentation</span></pre>
    `
  },
];
