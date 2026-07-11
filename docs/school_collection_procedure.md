# School data-collection procedure (Vahini pen, step by step)

A practical, beginner-friendly guide for collecting English character data
from school children with the Vahini pen. It covers: the setup, how the
pre-printed A4 sheet must look, how ground truth works when the laptop screen
is the only instructor, exactly how long everything takes, how to store the
recordings, and how to train afterwards. Read
`docs/vahini_dataset_collection.md` for the full protocol; this document is
the classroom fieldwork manual. Before the first school visit, complete
`docs/dry_run_checklist.md` end to end.

## The whole method in 8 lines

1. `python tools/make_sheets.py --students S0001 ...` prints 7 A4 sheets per
   student, each box numbered, AND writes the matching prompt CSV.
2. The capture program plays that same CSV: screen says "Box 12 — write g".
3. The child writes **g** in box 12 on the paper. The pen streams 16 channels
   over USB-C; the program records raw.
4. On pen-lift the screen shows the reconstructed stroke for 1.5 s (fun +
   sanity check). Enter/auto = next box, R = redo.
5. The saved recording is labeled `g` automatically — the prompt IS the
   ground truth. Nobody labels anything by hand.
6. ~310 boxes ≈ 45 min per student (two blocks for young children).
7. Everything lands in `data/raw/<school>/<student>/<session>/`;
   `tools/build_dataset.py` turns it into training files.
8. The ink on paper stays as the audit trail: scan 5% later and check the
   boxes match the labels.

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

**Don't design the sheets by hand** — generate them:

```bash
python tools/make_sheets.py --students S0001 S0002 S0003 --out sheets/
# sheets/sheets_S0001.pdf          -> print this (laser, 100% scale, no fit-to-page)
# sheets/sheets_S0001_prompts.csv  -> the capture program plays exactly this list
```

The PDF and the CSV come from the same seeded shuffle, so screen and paper
cannot disagree. Print at 100% scale and staple in order.

## 3a. The capture application (`tools/capture_app.py`)

The screen program is implemented in this repo — Tkinter with an embedded
matplotlib canvas for the reconstruction:

```bash
# real pen (USB-C serial; firmware sends one CSV line per sample: counter,ch1..ch16)
python tools/capture_app.py --student S0001 --school school01 \
    --prompts sheets/sheets_S0001_prompts.csv --pen-serial VP-003 \
    --port /dev/ttyACM0 --fullscreen

# no hardware yet — the simulator "writes" the prompted character itself,
# so the whole flow can be developed and rehearsed today:
python tools/capture_app.py --student S0001 --school school01 \
    --prompts sheets/sheets_S0001_prompts.csv --pen-serial SIM --simulate
```

What it does, in order: **calibration** (5 s pen-at-rest → per-channel bias
and noise floor saved to `calibration.json`; the pen-down force threshold
and the QC noise gate are derived from this measurement, not guessed) →
practice row → prompt loop. Each accepted sample passes QC (duration
0.3–8 s, real force activity, accel energy above the calibrated noise
floor, no sample-counter gaps); failures show a child-friendly retry
message. After each pen-lift the stroke is **reconstructed** (bias-removed
front-accel double integration with zero-velocity endpoint correction,
PCA onto the writing plane, then rotated upright against the prompted
glyph — see `docs/examples/reconstruction_demo.png`) and shown for 1.5 s;
Enter/auto = next, R = redo, B = extra break, Esc = stop (resumes later).
Keys are operator-facing; children only write. Recordings, `labels.csv`,
`meta.json`, and calibration land exactly in the §6 layout.

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

**45 minutes is too long for a child in one stretch — the session is broken
into blocks, and the app enforces this itself** (`tools/capture_app.py
--break-every 70 --break-secs 120`):

| Block | Boxes | Writing time | Then |
|---|---|---|---|
| warm-up | practice + 1–70 | ~9 min | 2-min break |
| 2 | 71–140 | ~8 min | **5-min big break** |
| 3 | 141–210 | ~8 min | 2-min break |
| 4 | 211–280 | ~8 min | 2-min break |
| 5 | 281–310 | ~4 min | done ⭐ |

**What the child does in a break** (the app's break screen says exactly
this, with a countdown): put the pen in the tray (never let it dangle by
the cable), stand up, shake hands and fingers out, stretch tall, look at
something far away. The operator uses the same 2 minutes to glance at the
cable routing and the sheet position. Nobody unplugs anything.

**Break inside a break** (tired hand, toilet, lost focus): any time, the
child just stops — the app sits waiting for the next pen-down forever, no
timeout. The operator can also press **B** for an immediate break screen,
or **Esc** to end the sitting entirely: the session resumes at the next
empty box when the child comes back, even on another day (sheets 1–4 today,
5–7 tomorrow is completely fine).

**If 45 min is still too much** (younger classes): generate sheets with
`--reps 3` instead of 5 → 186 boxes ≈ **25–30 min including breaks**. The
accuracy cost is small — writer *count* matters far more than repetitions
per writer — so prefer more students at 3 reps over fewer students at 5.

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

### 6a. Five pens at once — clubbing the stations' data after each session

When all 5 pens run simultaneously, each station laptop holds its own
`data/raw/` tree. They stay separate all day; **merging happens once, at the
end of the session**, on the master laptop/drive:

1. **During the day, never share folders between stations.** Each laptop
   only ever writes its own tree. The thing that makes merging trivial later
   is discipline *now*: every student code is unique (assigned from the
   class roster before the session — codes are pre-printed on the sheets),
   and one student writes at exactly one station.
2. **End of session**: copy each laptop's `data/raw/` onto the master drive
   (USB stick or shared folder), named per station:
   `station1_raw/ ... station5_raw/`.
3. **Merge with the tool** (never by hand-dragging folders):

   ```bash
   python tools/merge_raw.py \
       --stations /media/usb/station1_raw /media/usb/station2_raw \
                  /media/usb/station3_raw /media/usb/station4_raw \
                  /media/usb/station5_raw \
       --master data/raw
   ```

   Because every session is keyed by `school/student/session`, merging is a
   verified copy. The tool: skips sessions already merged (safe to re-run),
   **verifies every copied file**, refuses and reports if two stations claim
   the same student+session with different content (= a student code got
   reused — resolve by hand, don't guess), appends every merge to
   `data/raw/merge_log.csv`, and prints **samples-per-pen counts** so you
   can watch the §11 pen-rotation balance after every single session.
4. Then the usual evening routine on the merged master: backup to two
   places, `tools/build_dataset.py`, and log the counts in the tracking
   issue.

The five student sessions "club" into one dataset automatically at training
time — `build_dataset.py` walks the merged tree; each student's samples
carry their own writer code and (via `meta.json`) their pen serial, so
writer-independent splits and per-pen analyses need no extra work.

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

## 11. The 5-prototype pen fleet: rotation plan

You have 5 prototype pens. Prototypes differ slightly (sensor mounting,
bias, force response), and a model can learn to recognize *the pen* instead
of *the writing*. The defense is rotation + bookkeeping:

1. **Fingerprint every pen first** (`docs/dry_run_checklist.md` Phase A) and
   re-check each collection morning. A deviant pen is repaired or benched.
2. **Log the pen serial on every recording and every sheet header** (already
   in the metadata spec). This costs nothing and makes every analysis below
   possible.
3. **Break the pen↔class correlation.** Never let one pen collect one whole
   class or one whole school by itself. With 3 stations per visit, take 3 of
   the 5 pens and *change which three* on every visit (e.g. day 1: pens
   1/2/3, day 2: pens 4/5/1, day 3: pens 2/3/4 …). Within a class, students
   are assigned to stations arbitrarily — that is enough randomization.
4. **Rotate within the day too**: swap the pens between stations at the
   midday break, so morning/afternoon effects don't attach to one pen.
5. **Hold one pen out per fold.** When freezing the 5-fold splits, build one
   extra evaluation: train on pens {1,2,3,4}, test on pen 5 (rotate). This
   is the device analog of writer-independence and is the honest measure of
   "will a *new* production pen work?" — with 5 prototypes you are one of
   the few groups that can even measure this.
6. **Per-pen normalization at training time**: standardizing per channel
   *per pen serial* (statistics from that pen's calibration recordings)
   removes most inter-prototype bias before the model ever sees it.

Rule of thumb for the assignment sheet the operator carries: **every pen
should end the study having recorded students from every school, every age
band, and both hands.** If a pen's metadata histogram looks different from
the fleet's, the rotation failed and that pen's data needs a closer look
before release.
