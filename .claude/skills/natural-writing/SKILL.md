---
name: natural-writing
description: Use when writing or editing any prose this repo ships - docs, READMEs, code comments, commit messages, PR bodies, issues, UI strings, printed copy, captions. Use when asked to "write like a human", "remove AI-isms", "sound natural", or when text will be published where machine-sounding writing would cost us credibility.
---

# Natural writing

Write like a specific person with opinions, not like the average of the
internet. Models drift toward the mean: generic praise instead of sharp
facts, abstract significance instead of concrete detail. This skill bans the
patterns that make that drift visible.

Be specific and plain. Say less, mean more. Pick one precise detail over
three vague adjectives. Trust the reader; don't explain why something matters
after every sentence.

Stripping the patterns can leave text that is clean and dead. That reads as
machine-written too. Keep the voice:

- Have opinions. React to facts, don't only report them. "I'm not sure this
  works" beats a neutral list of pros and cons.
- Vary the rhythm. Short sentences. Then a longer one that takes its time.
- Allow mixed feelings. "Impressive, and a bit odd" beats "impressive".
- Use "I" when it fits. First person is direct, not unprofessional.
- Be specific about the feeling. Not "this is concerning" but "a station
  loading the wrong prompt file and nobody noticing for a week bothers me".

## Words to kill

Never in the figurative or filler sense. Each is heavily overrepresented in
post-2023 machine text.

| Kill | Use instead |
|---|---|
| delve, delve into | explore, examine, dig into, look at |
| tapestry (figurative) | mix, combination, variety |
| underscore (verb) | show, reveal, prove |
| pivotal | important, major, key moment |
| intricate, intricacies | complex, detailed, tricky |
| foster, fostering | build, encourage, grow |
| garner | get, earn, attract, win |
| showcase | show, display, demonstrate |
| landscape (figurative) | field, scene, world, space |
| testament (to) | proof, sign, evidence |
| vibrant | lively, busy, active |
| crucial | important, critical, necessary |
| enhance | improve, boost, strengthen |
| enduring | lasting, long-running, persistent |
| interplay | interaction, tension, relationship |
| Additionally, (opening a sentence) | Also, / On top of that, / just start |
| align with | match, fit, follow |
| valuable insights | useful findings, what we learned |
| evolving landscape | changing field, shifting ground |
| indelible mark | lasting effect |
| deeply rooted | long-standing, ingrained |
| nestled | located, sitting, tucked |
| groundbreaking (figurative) | new, original, first |
| renowned | well-known, famous |
| boasts a | has |
| in the heart of | in, in central, in downtown |
| diverse array | range, mix, variety |
| breathtaking | striking, impressive |
| commitment to | focus on, effort toward |
| seamless, seamlessly | smooth, easy, without friction |
| robust, comprehensive, cutting-edge | say what it actually does |
| leverage (verb), utilize | use |
| streamline, supercharge, elevate, empower, unlock | say the specific thing |

## Sentence patterns to kill

**Significance inflation.** Never attach broader meaning to an ordinary fact.

    BAD   This etymology highlights the enduring legacy of the community's
          resistance and the transformative power of unity.
    GOOD  The Spanish colonizers changed the spelling to Bacnotan.

**Trailing -ing editorializing.** Don't end a sentence with a participial
phrase that tells the reader how to feel.

    BAD   The station supported express trains, contributing to the
          socio-economic development of the region.
    GOOD  The station ran express trains to Delhi, Patna, and Kolkata.

**Dressed-up copulas.** Use "is" and "are".

    BAD   Gallery 825 serves as LAAA's exhibition space.
    GOOD  Gallery 825 is LAAA's exhibition space.

**Theatrical reframing.** Drop "not only X, but Y". State the fact.

    BAD   It constitutes not only a work of self-representation, but a
          visual document of her obsessions.
    GOOD  It's a self-portrait that maps her obsessions.

**Rhetorical triples.** Don't pad to three items for rhythm.

    BAD   keynote sessions, panel discussions, and networking opportunities
    GOOD  talks and panels

**Challenges-then-optimism.** Never use the formula.

    BAD   Despite its success, the canal faces challenges including...
          Future investments could enhance its efficiency.
    GOOD  The canal silts up above the third lock. Dredging is budgeted
          for March.

**False ranges.** "From X to Y" only when X and Y are ends of a real scale.

    BAD   from scientific discovery to artistic expression
    GOOD  in science and art

**Elegant variation.** Repeat the noun. If you said "mentors", say "mentors"
again, not "facilitators", then "field staff".

**Vague attribution.** Name the source or cut the claim.

    BAD   Experts believe it plays a crucial role.
    GOOD  A 2019 survey by the Chinese Academy of Sciences found three
          endemic fish species in the river.

**Padding.**

| Kill | Replace with |
|---|---|
| In order to | To |
| Due to the fact that | Because |
| At this point in time | Now |
| In the event that | If |
| It is important to note that | just say the thing |
| has the ability to | can |
| a large number of | many |
| in terms of | rephrase or cut |

**Over-qualifying.** "It could potentially possibly be argued that the policy
might have some effect" is "The policy may affect outcomes".

**Performative agreement.** Not "Great question! You're absolutely right".
Respond to the substance.

**Cheerleading endings.** Not "exciting times lie ahead". End with a
specific: "Two more schools are booked for March."

## Formatting

- Headings in sentence case, not Title Case.
- Bold marks the one thing not to miss. Not "key takeaway" decoration.
- Prose when a paragraph works. No bold-header-colon bullet lists.
- Tables only when the data is genuinely tabular.
- No emoji decorating headings or bullets.
- Straight quotes and apostrophes in code, docs, commits and issues.
  Typeset print copy and rendered UI may use typographic quotes where that
  is correct typography; that is not an AI tell.
- **Em dashes: default to none.** Use a comma, a colon, or a new sentence.
  One or two in a long piece is the ceiling, never the habit.

## Never narrate your own trustworthiness or your own edits

Shipped text states what is true. It does not comment on its own candour, and
it does not talk about earlier versions of itself. The reader wants the
result, not a confession.

    BAD   Against the literature that is +4.4, not the larger margin this
          README used to claim.
    GOOD  CNN+BiLSTM (Ott et al., ACM MM 2022, Table 3) reports 68.06%.

    BAD   To be frank, the numbers here are single-seed.
    GOOD  Single seed, fold 0.

    BAD   I should flag the part that does not flatter us.
    GOOD  26-class: 78.5 lower against their 79.48.

Specific bans:

- "honest", "honestly", "to be fair", "I should flag", "worth being clear"
- "this used to say X", "previously claimed", "corrected from", "I was wrong"
- "not the larger margin", "which does not flatter us", "the part I got wrong"
- Any sentence whose subject is the document, the author, or a past revision.

A number with its conditions stated needs no character reference. Adding one
reads as protesting too much. Put corrections in the commit message, where the
history belongs.

## Keep PRs and issues to what the reader must act on

A PR body says what changed and what a reviewer must check. An issue says what
to do and how to tell when it is done. Neither is a place to reproduce the
analysis, restate the reasoning, or narrate the process of getting there.

- No section explaining what you tried and rejected unless it changes the
  review.
- No restating numbers already visible in the diff or the linked doc. Link.
- No "caveats" section rehearsing what the docs already say.
- Detail belongs in the code, the docs, or the commit message. A PR body that
  runs past a screen is usually a doc in the wrong place.

Add sections only when asked.

## Things never to add

- "Challenges and Future Outlook" sections.
- Closing summaries: "In summary", "In conclusion", "Overall".
- Disclaimers: "It's important to note", "It's worth mentioning".
- Notability padding: "featured in prominent outlets".
- Speculation about what isn't documented: "details are not widely
  available". Say what we checked and what we found.
- Chatbot artifacts: "I hope this helps", "Let me know if", "Would you
  like".

## Self-check before committing

Two passes.

Pass 1, patterns. Ask: what makes this obviously machine-written? List the
tells that remain. Fix them.

Pass 2, voice. Ask: does this sound like a person wrote it, or like nobody
wrote it? Clean but lifeless needs voice added back: an opinion, a specific
reaction, a varied rhythm.

Then four questions:

1. Could this paragraph apply to fifty other subjects by swapping the noun?
   If yes, add a fact only this subject has.
2. Does any sentence tell the reader how to feel about a fact? Cut it.
3. How many banned words are left? Replace all of them.
4. Read it aloud. Press release, or someone explaining it over coffee?

Checklist:

- [ ] No banned vocabulary
- [ ] No trailing -ing editorializing
- [ ] No "serves as" or "stands as" where "is" works
- [ ] No "not just X but Y"
- [ ] No significance inflation on ordinary facts
- [ ] No challenges-then-outlook structure
- [ ] No vague attribution; sources named or claims cut
- [ ] No filler phrases, no stacked hedges
- [ ] No flattery, no sales tone
- [ ] No chatbot artifacts
- [ ] Sentence-case headings, no emoji decoration
- [ ] Straight quotes, minimal bold, no em dashes
- [ ] Specific details, not generic praise
- [ ] Has a voice
- [ ] No narration of the document's own candour or its earlier revisions
- [ ] PR/issue body limited to what the reader must act on

## In this repo

imu2text ships a research README, the FAQ, `docs/`, and citations. The
reader is a researcher deciding whether to trust the numbers.

Research writing has its own AI tells beyond the general list:

- **Never inflate a result.** Report the split, the number of writers, and
  the metric. "64.8% writer-independent on the bundled subset" is a claim.
  "State-of-the-art accuracy" is not.
- **Never invent a citation or a dataset statistic.** If the PDF is not in
  `papers/`, say the number is second-hand and name where it came from.
  `papers/README.md` is generated from DOI lookups and mis-attributes
  several files, so it is not a source for authorship.
- **Attribute borrowed design, never borrowed code.** Citing a paper's
  architecture choice is fine. The implementation is ours, written
  independently. This matters more for AI-assisted changes, because a model
  can reproduce code it saw in training without anyone noticing.
- Don't pad a limitations section into the challenges-then-optimism shape.
  State what the model cannot do and stop.
