# Pre-field-trial dry-run and reconstruction-check procedure

Complete ALL of this before the first school visit. Nothing here needs a
child: phases A–C use the team; only after every gate passes does the
10-student pilot happen (`docs/school_collection_procedure.md` §9). With 5
prototype pens, phases A and B run **per pen**.

## Phase A — bench check ("pen fingerprint"), per pen, ~20 min

Record every step with the normal capture program; store the recordings under
`data/bench/<pen_serial>/<date>/`. A pen passes only if all six checks pass.

| # | Check | How | Pass criteria |
|---|---|---|---|
| A1 | Sample rate | Pen at rest on desk, record 60 s | rows / duration = 208 ± 2 Hz; sample counter has zero gaps |
| A2 | Noise floor & bias | Same 60 s recording | per-channel std and mean within the fleet median ± 3×; no flat (dead) channel |
| A3 | Accelerometer | Hold pen still in 6 orientations (each axis up, then down), 5 s each | gravity (≈9.81 in the pen's units) appears on the correct axis with the correct sign, others ≈ 0 |
| A4 | Gyroscope | Rotate the pen a full 360° about each axis, slowly | integrated angular rate per axis = 360° ± 10% |
| A5 | Force sensor | Press tip on a kitchen scale at ~50 g, ~150 g, ~300 g | force channel is monotonic in the applied weight; returns to rest value within 1 s of lift |
| A6 | Magnetometer | Slowly rotate pen horizontally through 360° | field magnitude roughly constant (earth field), direction sweeps a full circle; no spikes near the USB cable |

Then compare **across the 5 pens**: for each channel, plot the five rest-noise
stds and biases side by side. Any pen whose channel is an outlier (>3× the
fleet median) is repaired or excluded — a deviant prototype quietly poisons
the dataset. Keep the fingerprints; re-run A1/A2 each collection morning
(2 min per pen) to catch drift or transport damage.

Record A5's typical rest-vs-writing force values — the capture program's
pen-down threshold comes from this measurement, not from a guess.

## Phase B — reconstruction validation, per pen, ~15 min

The on-screen reconstruction is a QC gate and the child's feedback, so it
must be *recognizably right*, not pixel-accurate. Validate it blind:

1. An adult writes a fixed script of 20 shapes in sheet boxes: circle,
   square, triangle, zigzag, figure-8, spiral, and the letters
   `A o l x S 3 7 g M w e k` plus two personal signatures.
2. A **second** team member, who did not watch the writing, is shown only the
   20 reconstructions and names each shape/letter.
3. **Pass: ≥16/20 named correctly per pen.** Below that, the reconstruction
   (filtering, integration, drift correction) needs work before the field —
   children will see "wrong" shapes for good writing and start distrusting or
   redoing everything.
4. Also check while writing: reconstruction appears **< 1.5 s** after pen-up;
   stroke count matches reality (the force channel splits strokes correctly —
   write `i`, `t`, `=`, which have pen lifts, and confirm 2 strokes each);
   writing fast vs. slow both reconstruct.

Log the score per pen in the bench folder (`reconstruction_check.csv`:
pen, date, score, examiner).

## Phase C — end-to-end rehearsal (adults playing students), ~2 h once

Two adults each run a **complete, unabridged** student session on the real
rig: printed sheets from `tools/make_sheets.py`, USB-C routing, calibration,
practice row, all 310 boxes, deliberately including these events:

- [ ] press **R** (redo) at least 5 times → old recording replaced, redo_count logged
- [ ] write with the **wrong pressure** once (too light) → program asks to retry, sample not saved as ok
- [ ] **unplug the USB-C mid-session**, replug → program resumes at the next empty box, no file corrupted
- [ ] deliberately **skip a box on paper** and follow the screen anyway → verify afterwards which box the ink landed in vs. the labels (this is the desync drill; the screen must always win, and the audit must catch it)
- [ ] one **left-handed** run (or right-hander simulating): cable mirrored, no cable drag on the force trace
- [ ] finish sheet swap between sheet 4 and 5 without operator help

Then the data side, same day:

- [ ] `labels.csv` has 310 `ok` rows + practice rows marked `practice`
- [ ] `python tools/build_dataset.py --raw data/raw --out /tmp/rehearsal` runs clean; sample count = 620 (2 adults × 310)
- [ ] `python onhw_models.py --channels 16 --imu-file ... --writers-file ... --split random --epochs 2` completes (accuracy is irrelevant; the plumbing is the test)
- [ ] backup script copies the raw tree to both destinations; restore ONE file from the backup and diff it
- [ ] time the sessions: recompute the §5 time tables in the procedure doc with the measured seconds/sample

## Phase D — go / no-go gate

Proceed to the 10-child pilot only when every line is true:

- [ ] 5/5 pens pass Phase A; fingerprints archived; morning re-check script ready
- [ ] 5/5 pens score ≥16/20 in Phase B
- [ ] Phase C completed with zero unexplained files, zero label mismatches
- [ ] consent forms signed for the pilot class
- [ ] pen-rotation plan for the visit written down (which pens at which stations — see procedure doc §11)
- [ ] printed sheets + prompts CSVs generated for every pilot student code, spot-checked against each other (box 1 on paper == box 1 in CSV for 3 random students)

If any pen fails and cannot be fixed, run the field trial with fewer pens —
never with a pen that failed the bench check.
