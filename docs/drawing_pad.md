# The drawing pad: two AIs in a row

The playground's "Draw it" pad is a simulation. The product is a pen with
motion sensors, and the AI this repo trains reads that pen's movement. A
finger or a mouse on a screen has no sensors, so the pad cannot hand its
drawing to that AI. It runs two separate AIs instead, one after the other,
and every number on this page is measured for that chain, not for either AI
alone.

Expect mistakes from the pad that a real pen would not make, and the other
way round. The only test of the product is a real pen in a real hand.

## The pipeline

```text
your drawing (positions and times)
  │
  ├── movement: speed, acceleration, turning, pen up/down   (movement.js)
  │     shown to you; not read by any AI
  │
  └── AI 1, the drawing reader: the shape → a character      (reader.js)
        │   trained on UJI Pen Characters, runs in your browser
        │
        └── a real recording of that character by another writer
              │   OnHW, made with the sensor pen, 13 signals at 100 Hz
              │
              └── AI 2, the pen AI (CNN-BiLSTM): the movement → a character
                    its answer was computed in advance, not in your browser
```

Each step's answer is shown on the page, so a mistake can be traced to the AI
that made it.

## AI 1: the drawing reader

**What it reads.** The shape of the drawing, as a 28 x 28 image. Strokes are
drawn onto the grid the same way in training (`scripts/drawing_reader.py`,
`rasterize`) and in the browser (`playground/reader.js`), so stroke order,
direction and count do not matter.

**Data.** UJI Pen Characters v2 (Prat, Castro, Llorens, Marzal and Vilar,
2008, doi:10.24432/C5FG8S, CC BY 4.0): pen trajectories from 60 writers. The
published split puts 40 writers in training and 20 in test. Validation comes
from 6 of the 40 training writers; the 20 test writers are scored once, at
the end. The data is not in this repository.

We first planned EMNIST and did not use it: its licence terms are not clear
enough to ship weights trained on it. UJI's are.

**Training.** Each training drawing gets 24 copies, each rotated (up to 15
degrees), slanted, stretched, shaken slightly and drawn with a different
line width, so the reader sees more than 34 people's handwriting.

**Model.** Three 3 x 3 convolutions (16, 32 and 48 filters) with max pooling,
one hidden layer of 96, and an output of 62 classes: 0 to 9, A to Z, a to z.
66,254 weights, stored as 8-bit integers with one scale per tensor (88 KB in
`playground/reader-weights.js`). The forward pass is plain JavaScript and
takes a few milliseconds.

```bash
python -m scripts.drawing_reader data/ujipenchars2/ujipenchars2.txt \
    --out results/drawing_reader --export playground/reader-weights.js
```

**Accuracy**, UJI published test split, 20 unseen writers, 2,480 drawings,
62 classes, seed 0, one training run (not deterministic), from
`results/drawing_reader/summary.json`:

| What | Right |
|---|---|
| Reader alone, all 62 classes | 78.1% |
| Reader alone, capitals and small letters counted as one | 89.6% |
| The hand-drawn templates it replaced, same drawings | 30.2% |

On the page the reader knows the task the visitor chose, and that helps. On
the Numbers task it picks among digits only; on Letters, among letters. Same
test drawings, placed on the pad as the page receives them, from
`results/drawing_reader/pipeline.json` (`scripts/drawing_pad_eval.py`):

| On the page | Right |
|---|---|
| Digits, Numbers task (400 drawings) | 96.2% |
| Letters, Letters task (2,080 drawings) | 76.3% |
| Letters, capitals and small letters counted as one | 90.2% |

Most letter errors are a letter read as its other case: o and O, s and S,
c and C look the same as an image, and only their size next to other letters
tells them apart.

**Several characters.** "12" or "108" is split into characters before
reading. `splitWithReader` in `playground/shapes.js` tries every way of
grouping the strokes in writing order, scores each group with the reader,
and keeps the grouping that reads best. A clear gap between strokes is
always a cut. Sequences were built from one test writer's characters side
by side, from slightly overlapping to well apart (seed 7, 200 of each):

| Sequence | Read exactly |
|---|---|
| 2 or 3 digits, Numbers task | 94.5% |
| 2 letters, Letters task, capitals and small letters counted as one | 82.5% |
| 2 letters, exact case | 56.0% |

1.2% of the 2,480 single characters were wrongly split in two.

## AI 2: the pen AI

**What it reads.** The movement of the sensor pen: 13 signals at 100 Hz
(two accelerometers, a gyroscope, a magnetometer, tip force). It never sees
a drawing.

**Data.** OnHW (Ott et al., Fraunhofer IIS and STABILO), recorded with the
sensor pen. For non-commercial use, so neither the data nor weights trained
on it are in this repository.

**Model.** A CNN-BiLSTM with attention (`imu2text/models.py`). The letters
answer averages five runs from different seeds.

**Accuracy**, writer-independent, from `docs/benchmarks.md`:

| Task | Right |
|---|---|
| Letters, 52 classes, 5-seed ensemble, official right-handed test (7,956 recordings) | 74.46% |
| Same, capitals and small letters counted as one | 86.14% |
| Numbers and symbols, 15 classes, seed 0 (611 recordings) | 71.03% |

The published result for comparison is 68.06% (Ott et al., ACM MM 2022,
Table 3).

## How the two combine

The chain is right only when both AIs are right. Two effects matter.

**A reader mistake sends the pen AI the wrong recording.** Draw a small p,
let the reader say P, and the pen AI reads a recording of P. Whatever it
answers, it was asked the wrong question. The page shows the reader's answer
first, so the visitor sees which AI went wrong.

**Which recording is shown.** For each character the page uses one recording:
the median of that class by the pen AI's confidence in the right answer
(`one_per_class` in `scripts/build_playground.py`). That recording is read
right whenever the pen AI reads that character right more often than not.
On those recordings it is right for 46 of 52 letters and all 15 numbers and
symbols, which is better than its 74.46% and 71.03% over all recordings. The
page must not let that look like the pen AI's accuracy.

UJI test drawings through the whole chain (20 unseen writers, reader seed 0,
pen AI as above, `results/drawing_reader/pipeline.json`):

| | Reader right | Both right, the page's recording | Both right, expected on any recording |
|---|---|---|---|
| Digits, Numbers task | 96.2% | 96.2% | 67.0% |
| Letters, Letters task | 76.3% | 69.3% | 58.4% |

"Expected on any recording" multiplies each correct reader answer by the pen
AI's measured rate for that character on unseen writers. It is the honest
figure for "draw a letter, get the right answer from a real recording". The
page shows that rate for each character next to the pen AI's answer
(`class_accuracy` in `data/public.js`).

## What limits the cost of a wrong read

- **The task narrows the answer.** On Numbers the reader picks only among
  digits, which is why digits reach 96.2%.
- **"Not sure" instead of a guess.** Below a minimum confidence the reader
  asks for another stroke rather than send the pen AI anything.
- **The reader's three best guesses are shown** with their scores, so a
  close call is visible.
- **"Read as … instead".** When a drawing reads two ways (10 or "lo"), the
  other reading is one click away.
- **Each step is labelled.** The visitor sees the reader's answer before the
  pen AI's, and the pen AI's answer is marked as coming from a recording by
  another writer.

## What the pad cannot show

- How the pen AI does on the visitor's own handwriting. Only a sensor pen
  can test that.
- Tilt, tip force and the pen's rotation. A screen gives none of them; the
  movement shown under the pad is worked out from positions and times only.
- Whether the drawing reader's 78% would hold on children's handwriting or
  on other scripts. UJI is adult Latin handwriting.

## Files

| File | What |
|---|---|
| `scripts/drawing_reader.py` | trains and scores the drawing reader, exports the weights |
| `scripts/drawing_pad_eval.py` | scores the pad as the page runs it, and the two AIs in a row |
| `results/drawing_reader/` | its scores and test predictions |
| `playground/reader.js`, `reader-weights.js` | the reader in the browser |
| `playground/shapes.js` | splits a drawing into characters, applies the task, dots and lines by rule |
| `playground/movement.js` | speed, acceleration and turning from the drawing |
| `scripts/build_playground.py` | the pen AI's saved answers, one recording per character |
| `tests/test_playground_shapes.py`, `test_playground_movement.py` | run the above under Node |
