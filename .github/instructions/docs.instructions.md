---
applyTo: "**/*.md"
---

# Docs review: imu2text

Docs follow `.claude/skills/natural-writing/SKILL.md` and the Writing section
of `AGENTS.md`. `tests/test_docs_style.py` already fails on em dashes, spaced
hyphens used as dashes and the worst kill-list words, so review for what a
script cannot see.

## Check every Markdown change for

- **A home.** Does this belong in an existing page? One topic, one doc. A new
  file needs a reason, and a link from `README.md` or `docs/getting_started.md`.
- **Numbers with their conditions.** Split, dataset, class count, seed, and
  the file in `results/` or the run it came from. A number copied from another
  doc should be a link instead.
- **Length.** The README stays one screen. A paragraph that restates a table,
  explains why something matters, or summarises the page is cut.
- **Plain sentences.** "is" instead of "serves as", no significance
  inflation, no trailing "-ing" clause telling the reader how to feel, no
  "not only X but Y", no closing summary.
- **No narration.** A doc states what is true now. It does not mention its
  earlier versions or vouch for its own candour; corrections go in the commit
  message.
- **Citations.** Papers are cited by author, venue and year. No private or
  reference repository is named or linked. No PDFs, datasets or weights are
  committed.
- **Links that work.** Relative links point at files that exist.
- **Sentence-case headings, no emoji.**

A PR body says what changed and what to check, in under a screen.
