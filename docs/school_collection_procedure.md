# School data-collection procedure (Vahini pen, step by step)

A practical, beginner-friendly guide for collecting English character data
from school children with the Vahini pen. It covers: the setup, how the
pre-printed A4 sheet must look, how ground truth works when the laptop screen
is the only instructor, exactly how long everything takes, how to store the
recordings, and how to train afterwards. Read
`docs/vahini_dataset_collection.md` for the full protocol; this document is
the classroom fieldwork manual.

## 1. The recording rig

- **Pen**: Vahini pen, **16 sensor channels at 208 Hz**, in this fixed order:

  | Channels | Sensor |
  |---|---|
  | 1 | force (tip) |
  | 2–4 | front accelerometer x, y, z |
  | 5–7 | front gyroscope x, y, z |
  | 8–10 | magnetometer x, y, z |
  | 11–13 | rear accelerometer x, y, z |
  | 14–16 | rear gyroscope x, y, z |

  So: `force(1) + front_accel(3) + front_gyro(3) + magnetometer(3) +
  rear_accel(3) + rear_gyro(3) = 16`. (OnHW pens have 13 — no rear gyro — so
  the rear-gyro triplet goes last; dropping channels 14–16 gives an
  OnHW-compatible 13-channel view for transfer learning.)
- **Connection**: USB-C cable to the laptop. Leave generous cable slack and
  route the cable **away from the writing hand** (over the top of the desk,
  taped down at the desk edge). For left-handed children mirror the routing.
  A dragging cable changes the force/accel signals — this matters.
- **Desk**: child seated on a chair at a table, A4 sheet on a firm pad,
  laptop screen facing the child at arm's length behind/above the sheet.
- **Record raw**: the capture program writes exactly what the pen streams —
  no filtering, no scaling. Store host timestamp + the pen's own sample
  counter with every row so dropped packets are detectable (at 208 Hz the
  counter must increase by 1 every ~4.8 ms).

## 2. What one "sample" is, and the session plan

One sample = one character written once in one box. The English character
set is 26 uppercase + 26 lowercase + 10 digits = **62 characters**. Five
repetitions of each gives 310 ≈ **300 samples per student** — that is where
the "300 samples" target comes from.

## 3. The pre-printed A4 sheet (so children are never confused)

Children will write *anything* on a blank page, so the sheet does the
guiding. Print (laser, not inkjet — no smudging) sheets with:

```
 ┌──────────────────────────────────────────────────────────────┐
 │  VAHINI  •  Sheet 3 of 7  •  Student code: ____  Pen: ____   │
 ├──────────┬──────────┬──────────┬──────────┬──────────┬───────┤
 │ 97    ᵍ  │ 98    ᵀ  │ 99    ⁴  │ 100   ʲ  │ 101   ᴹ  │ 102 ˢ │
 │ ┈┈┈┈┈┈┈  │ ┈┈┈┈┈┈┈  │ ┈┈┈┈┈┈┈  │ ┈┈┈┈┈┈┈  │ ┈┈┈┈┈┈┈  │ ┈┈┈┈┈ │
 │ (child   │          │          │          │          │       │
 │  writes  │          │          │          │          │       │
 │  here)   │          │          │          │          │       │
 ├──────────┼──────────┼──────────┼──────────┼──────────┼───────┤
 │ ...  8 rows × 6 columns = 48 boxes per sheet ...             │
 └──────────────────────────────────────────────────────────────┘
```

Rules that make it work:

1. **One box per sample**, ~28×28 mm, 6 columns × 8 rows = 48 boxes per
   sheet; 310 samples = **7 sheets** per student, pre-stapled in order.
2. Each box shows its **box number** (top-left, bold) and the **target
   character small in the top-right corner** as a reference. Do NOT print a
   large gray letter to trace over — traced writing is unnaturally uniform
   and the model would learn tracing, not handwriting.
3. **Dotted four-line English ruling** inside each box (the same ruling used
   in school copybooks) so size and baseline are consistent.
4. The **laptop screen always shows the same box number and character** in
   large print: “Box 97 — write small letter g”. Child finds box 97 (numbers
   run left-to-right, top-to-bottom, sheets stapled in order), writes, done.
   Screen and paper can never disagree because the sheets are generated from
   the same prompt list the program plays back.
5. Print the target in the corner in a **school-style font** (e.g. a
   handwriting-education font), because children copy letterforms literally —
   a two-storey ‘a’ on screen produces two-storey ‘a’s on paper.
6. First row of sheet 1 is a **practice row** (5 boxes, marked “PRACTICE”);
   the program discards those recordings automatically.
7. Header of every sheet: student code (NOT the name), sheet number, pen
   serial, date. The paper is your permanent audit trail — box numbers let
   you re-check any recording against ink forever.

**Prompt order**: shuffle the 310 prompts once per student (fixed seed per
student code) instead of A,A,A,A,A,B,B,… — repetition boredom produces
sloppy repeats, and shuffling spreads fatigue evenly across classes.

## 4. Ground truth with no companion — the screen is the instructor

You do not need a human checker per child. The trust chain is:

1. **The label IS the prompt.** The program commands “Box 97 — g”, so the
   recording saved for box 97 is labeled `g`. Nothing is ever hand-labeled.
2. **Live reconstruction on screen.** After each pen-up, the Python program
   integrates/plots the stroke and shows the reconstructed shape next to the
   prompt for ~1.5 s. The child sees their own writing appear — this is the
   engagement trick AND the first QC gate. Two big on-screen buttons/keys:
   **Enter/space = looks right, next box** (or auto-advance after 1.5 s of
   pen-up idle), **R = redo** (recording replaced, redo count logged, child
   writes again in the SAME box → if a redo happens, the box will contain two
   attempts in ink; the log says which one counts — the last).
3. **Automatic checks per sample**, enforced by the capture program before
   accepting: force channel shows a real pen-down (else “I didn’t feel the
   pen — press harder and try again”), duration within 0.3–8 s, sample
   counter continuous, no flat/saturated channel. Failing samples trigger an
   automatic child-friendly redo message.
4. **Paper audit.** ~5% of sheets are scanned after the fact and compared
   with labels by an adult — this is how you *prove* the pipeline is honest
   without supervising every child.
5. A teacher stays in the room for order and consent reasons, but never
   needs to touch the labeling.

## 5. How long it takes (the math)

Per sample: children write a character in ~2–3 s; add prompt reading,
pen-up, the 1.5 s reconstruction display, and occasional redos → budget
**~7 s per sample** on average.

**Per student** (310 samples ≈ “300 samples”):

| Step | Time |
|---|---|
| Seat, connect USB-C, calibration recording (pen at rest 5 s + one spiral) | 3 min |
| Instructions + practice row | 3 min |
| 310 samples × 7 s | ~36 min |
| Sheet swaps, wiggle breaks, buffer | 4 min |
| **Total per student** | **~45 min** |

For children under ~10, split into **two 20–25 min blocks** (sheets 1–4,
then 5–7 later the same day or next day) — quality collapses when young
children write for 45 minutes straight. The program resumes at the next
empty box automatically.

**Per class of 30–40 students** (a “station” = one pen + one laptop + one
desk):

| Stations | 30 students | 40 students |
|---|---|---|
| 1 | 30 × 45 min ≈ 22.5 h → **4–5 school days** | 30 h → **6 school days** |
| 2 | ~11 h → **2–2.5 days** | 15 h → **3 days** |
| 3 | ~7.5 h → **1.5 days** | 10 h → **2 days** |

(assuming ~5 usable collection hours per school day; students rotate to the
station between lessons).

**Whole dataset** (protocol target 150–300 writers): 300 students × 45 min
= 225 station-hours → with 3 stations ≈ **15 school days ≈ 3 weeks** of
visits. With 150 students it is ~7–8 days. Budget one extra pilot day (one
class of 10) before everything — see §9.

## 6. How to store the samples

One folder per student per session; one CSV per sample; one manifest per
session. Nothing is ever overwritten or renamed.

```
data/raw/<school>/<student_code>/<session>/
    meta.json          # student code, age band, handedness, pen serial,
                       # firmware, sheet version, date, operator initials
    calibration.csv    # the rest+spiral recording from setup
    rec_00097.csv      # one file per box number (the accepted attempt)
    labels.csv         # manifest: box, label, redo_count, qc_status, rec file
```

`rec_*.csv` columns, fixed order (2 bookkeeping + 16 sensor channels):

```
host_ts_us, counter, force,
fax, fay, faz,   fgx, fgy, fgz,     # front accel, front gyro
mx,  my,  mz,                       # magnetometer
rax, ray, raz,   rgx, rgy, rgz      # rear accel, rear gyro
```

`labels.csv` columns: `box,label,rec,redo_count,qc` (qc is `ok`, `discard`
or `practice`).

Convert everything to the training format with the bundled tool:

```bash
python tools/build_dataset.py --raw data/raw --out data/vahini_v1
# -> data/vahini_v1/all_x_dat_imu.pkl  (list of (T,16) float32 arrays)
#    data/vahini_v1/all_gt.pkl         (list of character labels)
#    data/vahini_v1/writers.pkl        (list of student codes, same order)
```

Back up `data/raw/` to two places the same evening (external drive + cloud).
The raw tree is the dataset; pickles can always be regenerated.

## 7. How to train (novice commands)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# character models, writer-independent split from real writer IDs:
python onhw_models.py --channels 16 \
    --imu-file data/vahini_v1/all_x_dat_imu.pkl \
    --gt-file  data/vahini_v1/all_gt.pkl \
    --writers-file data/vahini_v1/writers.pkl \
    --split writer --augment 4 --rnn-units 100 --rnn-layers 2 --epochs 60

# learning curve + projection figures:
python make_learning_curve.py && python plot_results.py
```

Retrain after every school visit and extend the learning curve — the curve
tells you when adding more students stops paying.

## 8. Consent (children — do this first)

Parental/guardian **written consent before any child holds the pen**, a
child-friendly verbal explanation, and the right to stop at any time. Store
only student codes in the data; keep the code↔name list on paper with the
school, never in the repository. Handwriting dynamics are personal data —
treat the raw tree as confidential.

## 9. Pilot: one class of 10 before everything

Run 10 students end-to-end, then check: redo rate (>15% means the prompts or
sheet confuse children — fix wording/font), samples with QC failures, actual
seconds per sample (recompute the tables above with YOUR number), and train
on the pilot to verify the pipeline runs. Only then book the full schedule.

## 10. Reality check: what accuracy this dataset buys

Isolated characters from unseen child writers will land in the **70–85%**
range once you have 150–300 writers (children are *more* variable than the
adult OnHW writers). That is expected and correct — see
`docs/vahini_dataset_collection.md` §9: the 99% product number comes from
adding word context + lexicon decoding + per-user enrollment + a reject
option on top of exactly this character dataset. Phase 2 of collection
(same rig, same sheets, word prompts from the school's word lists) is what
unlocks it — the character phase you are planning here is the foundation
that trains the encoder and calibrates everything.
