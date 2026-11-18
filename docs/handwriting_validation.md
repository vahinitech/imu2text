# Handwriting measurement and validation research

Checked 18 November 2026. Use this reading list to choose measurements and
validation experiments, not to claim that Vahini detects a disorder. Recognition,
trace reconstruction and handwriting assessment are different targets.
For the datasets already used by imu2text, see [datasets](datasets.md).

## What to read first

| Study | Population and measurement | Available material | Use for Vahini |
|---|---|---|---|
| [Toffoli et al., 2026](https://doi.org/10.1371/journal.pdig.0001503) | 707 Northern Italian pupils, grades 1-8; 50 Hz IMU and tip force, ordinary paper | Derived task-level indicators and demographics, not raw streams or ink | Define movement measures and plan local validation |
| [Dui et al., 2021](https://doi.org/10.1109/BHI50953.2021.9508539) | 52 second/third graders; pen-on-paper compared with tablet | Paired indicators, usability and handwriting-speed scores | Test whether the recording device changes the task |
| [Lunardini et al., 2021](https://doi.org/10.1109/TIM.2020.3045838) | 43 healthy adults in three age groups; motion and force | Instrument-validation paper; no downloadable cohort dataset verified | Plan calibration, reference comparisons and repeatability |
| [Ott et al., 2020](https://doi.org/10.1145/3411842) | 119 writers, 31,275 isolated alphabet recordings; 100 Hz sensor pen | Raw OnHW-chars sensor recordings and recognition labels | Recognition benchmark, not child assessment norms |
| [Ott et al., WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Ott_Joint_Classification_and_Trajectory_Regression_of_Online_Handwriting_Using_a_WACV_2022_paper.html) | One right-handed writer; about 18 repetitions of 83 characters | Paired inertial/trajectory material and visual recordings | Small-scale trajectory-method comparison, not unseen-writer evidence |
| [Toffoli et al., EMBC 2025](https://doi.org/10.1109/EMBC58623.2025.11253922) | 50 Italian fifth graders, 353 cursive instances of one word | Reconstruction study; public raw-data release not verified | Separate reconstructed traces from real scans in evaluation |
| [Asselborn et al., 2018](https://doi.org/10.1038/s41746-018-0049-x) | 298 children including 56 with dysgraphia; paper over a digitizing tablet | Study of 53 features; no open dataset permission verified | Compare assessment design, not transfer reported accuracy |

## The 2026 sensor-pen dataset

The [Zenodo deposit](https://zenodo.org/records/19485897) contains
`NUM_B.xlsx`, `NUM_C.xlsx`, `UNO_B.xlsx`, `UNO_C.xlsx` and
`indicators_description.docx`. Rows describe subjects; fields include grade,
age, handedness, language background and BVSCO-3 z-score. B/C distinguish block
and cursive. The tasks repeat "UNO" or write number-words for one minute.

The paper derives 178 indicators across time, force, smoothness, tilt,
oscillation and frequency domains. It does not measure letter formation or
spatial layout. Its cross-sectional grade comparisons are not evidence of an
individual child's improvement; the authors explicitly say the pen cannot yet
be considered a screening tool. Excluding diagnosed disorders, regional
recruitment, class imbalance and cursive familiarity limit interpretation.

Francesca Lunardini is a paper co-author, though not listed among the deposit's
creators. The paper discloses related patents, including US 11,993,098 and
EP 4127884. Dataset permission does not settle implementation patent rights.

The [indicator definitions](https://zenodo.org/records/19485897/files/indicators_description.docx?download=1)
distinguish a stroke from a letter: a stroke is continuous paper contact.
They define a pause as an in-air interval longer than two seconds. A joined
cursive word can contain fewer strokes than letters; stroke rate must not be
labelled character speed.

## What the files and hardware actually expose

Inspection of `UNO_B.xlsx` found one sheet, `Sheet1`, with worksheet dimensions
of 702 rows including the header and 189 columns. This is not a verified
count of nonempty observations or unique participants. No explicit subject-ID
column was found in the header; do not infer cross-task identity from row order. The first ten fields are subject/context
fields; `ExecutionTime` and the remaining columns are derived indicators.
Examples include `StrokeNum`, `RelStrokeNum`, `meanOnsheet`, `Onsheet_CV`,
`meanInair`, `meanPause`, `PauseNum`, `meanPonsheet` and `P_CV`.
A row is not a timestamped sequence of IMU samples. Do not feed these tables
into the existing 13-channel OnHW sequence loader. The Zenodo file list has
no PCB, CAD, firmware or feature-extraction code; that does not prove no such
release exists elsewhere.

The [2021 hardware paper](https://air.unimi.it/retrieve/handle/2434/813023/1708225/09311200.pdf)
discloses LSM6DSM six-axis sensing, a Forsentek FC8E load cell, STM32L476,
BlueNRG-MS BLE 4.1, 1 MB flash and AD623RMZ/ADS1115I force electronics.
That is not the same description as the 2026 nine-axis/piezoelectric device.
Do not present the older bill of materials as the verified 2026 build.

Predecessor processing includes force baseline correction, BEADS denoising,
a calibrated clipping threshold, contact segmentation and a two-second pause
rule. It uses 4 Hz low-pass filtering for writing and a separate tremor
pipeline. These disclosures do not establish exact reproduction of all 178
later indicators. Vahini must calibrate thresholds and filters for its own
sensor, sampling rate and geometry rather than copy constants blindly.

## Related evidence and its limits

### Device agreement and repeatability

Dui's [institutional abstract](https://re.public.polimi.it/handle/11311/1188793)
compares usability and concurrent validity of tablet and pen tasks. Its
[deposit](https://zenodo.org/records/7684965) provides `Characterization.xlsx`
and `pen_tablet_data.mat`: participant fields, pen/tablet usability scores,
BVSCO-2 grapheme counts and computed indicators. These are not raw streams.
This is a useful device-comparison study, not permission to treat tablet norms
as pen norms.

Lunardini's [institutional paper](https://air.unimi.it/handle/2434/813023)
compares static/dynamic force, tilt and stroke segmentation against reference
instruments, then tests repeatability in healthy adults. The earlier
[EMBC 2020 study](https://doi.org/10.1109/EMBC44109.2020.9175999) uses intraclass
correlation and minimal detectable change. Borrow the validation questions,
not their thresholds: Vahini needs its own measurement-error estimates.

### Recognition and trajectory reconstruction

OnHW-chars supplies two accelerometers, gyroscope, magnetometer and force
channels. Its official writer-independent splits answer whether recognition
generalizes to held-out writers on that acquisition system. They do not test
Vahini hardware or handwriting quality. Keep the existing
[benchmark protocol](benchmarks.md) separate from assessment experiments.

The WACV study records IMU data at 100 Hz, tablet reference trajectories at
30 Hz and camera recordings at 60 Hz. The pen uses a Wacom EMR replacement
for the refill during reference acquisition. Its one-writer collection is
not a population validation or ordinary-paper deployment test.

The [EMBC 2025 abstract](https://pubmed.ncbi.nlm.nih.gov/41337235/) evaluates
reconstruction of cursive "uno" against scans using Google Vision OCR.
One word and one grade cannot establish general handwriting recognition.
OCR success is also not a direct geometric trajectory-error measure.
The study identifies sampling, tip-to-sensor geometry and in-air trajectory
as areas needing work. No Vahini result follows from this experiment.

### Tablet-based assessment

Asselborn's study uses a five-minute BHK copying task on paper affixed to a
Wacom tablet, capturing coordinates, pressure and tilt at up to 200 Hz.
Different tablet models were used for the two groups, with pressure
calibration. The typically developing group was not individually rated for
dysgraphia. These conditions belong with any discussion of its results.
A photo or IMU-only stream does not contain the same observations.

## Data rights and access

The following is an access check, not legal clearance for a product. Article,
data, code, model and patent rights are separate. Do not commit downloaded
papers, participant rows, raw recordings or model weights.

| Resource | Verified access or permission | Decision |
|---|---|---|
| Zenodo 19485897 | Open; `metadata.license.id` is `cc-by-4.0` in the [record API](https://zenodo.org/api/records/19485897), checked 18 November 2026 | Used by the prototype below; source files stay outside git and derived reports carry attribution |
| Zenodo 7684965 | Open; `metadata.license.id` is `cc-by-4.0` in the [record API](https://zenodo.org/api/records/7684965), checked 18 November 2026 | Candidate for device-comparison analysis, not currently loaded |
| OnHW | [Author-hosted dataset card](https://huggingface.co/datasets/FelixOtt/OnHW_datasets) states CC BY-NC 4.0 for OnHW; [Fraunhofer source page](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html) hosts dataset documentation | No product weights or commercial training without applicable written terms |
| WACV paired material | Download referenced by Fraunhofer; no separate commercial grant verified | Research reference only until archive-specific terms are checked |
| EMBC 2025 and IEEE 2021 cohorts | Paper/abstract accessible; raw data and reuse terms not verified | Do not assume article access grants participant-data access |
| Asselborn cohort | No open participant-data license verified in this review | Method reference only; do not scrape participant material |

Zenodo's rendered pages showed an empty license heading during this check;
use the API metadata rather than interpreting that display as no license.
The author-hosted card lists separate ICROW conditions, including a prohibition
on training or heuristically improving recognizers with its benchmark data.
Do not assume one OnHW archive license covers every third-party dataset.
No third-party implementation code was copied. The local SIP downloader below
fetches only the pinned feature-table deposit, not raw pen recordings.

## Proposed Vahini validation work

These are proposed experiments, not completed work or approved clinical claims.

1. Write a measurement contract for the actual device: firmware version,
   sensor positions, units, axis convention, timestamp clock, sample-loss
   handling, calibration and filtering. Do not mix measured force with a
   photo-derived pressure proxy.
2. Start with contact duration, lift duration, stroke count and force
   variability. Check pen-down segmentation against a synchronized reference;
   report errors, not only successful examples. Freeze the pause definition.
3. Repeat the same task across sessions and devices. Estimate agreement and
   measurement error before showing a child's trend as improvement. Separate
   true change from a firmware, refill, grip or writing-surface change.
4. For reconstruction, collect consented paired reference trajectories and
   scans. Split by writer before fitting anything. Measure geometric error
   separately from OCR character/word error and task completion.
5. For assessment, agree a protocol with qualified local evaluators. Include
   grade, language, script, writing style and handedness; use blinded manual
   scores. Choose sample size from the intended precision and target use,
   not from the convenient size of a foreign dataset.
6. Keep all sessions from a child on one side of a split. Fit preprocessing,
   feature selection and thresholds on training data only. Hold out schools
   or sites for an external check and report subgroup uncertainty.
7. Separate technical validity, usability and educational benefit. A reliable
   sensor score does not establish diagnosis, referral accuracy or the effect
   of coaching. Do not ship diagnostic labels from a research classifier.

Before an experiment begins, record its target, eligibility, consent process,
reference measure, primary metric, exclusion rules, split, acceptance threshold
and stop condition. Publish only aggregates permitted by the study consent.

## Running the feature-table prototype

`imu2text.sip` implements acquisition, schema checks, aggregate inspection and
an independent logistic-regression baseline against a majority-class dummy.
It does not reproduce the paper's model search, within-grade scaling, TabPFN,
feature-selection procedure or reported performance. It does not reconstruct
handwriting or compute the published indicators from raw signals.

Use a separate Python 3.10-3.12 environment. The smaller dependency file avoids
installing neural-network frameworks for a table analysis:

```bash
python -m venv .venv-sip
# Activate the environment using your platform's activation command.
python -m pip install -r requirements-sip.txt
python -m imu2text.sip download --directory data/sip-19485897
python -m imu2text.sip inspect data/sip-19485897/UNO_B.xlsx \
  --output results/sip-local/UNO_B-audit.json
```

The downloader accepts only the five v1 files with pinned publisher MD5 and
byte size. It records SHA-256, creators, DOI, CC BY 4.0 and modification status
in local `provenance.json`. Existing mismatched files fail rather than being
replaced. MD5 is the publisher's integrity check, not proof of authenticity;
the fixed HTTPS source and recorded SHA-256 support an auditable acquisition.
Original files remain outside git under `data/sip-*`.

All four workbooks were downloaded and parsed, not just mocked. Observed rows:
UNO_B 701, UNO_C 680, NUM_B 700, NUM_C 674; each has 189 columns. The first ten
columns contain context or target values. `ExecutionTime` is excluded from the
model, leaving 178 predictors. There are no missing values among those
predictors in these files. UNO_C has one missing Z and NUM_C has two. Inspection
counts these; fitting excludes them and never imputes a target. First-grade
cursive rows are also excluded, with separate counts in the report.

### Identity and leakage controls

The published files provide no explicit subject ID. Do not join task tables
by row number or claim an unseen-writer test from a random row split.
The default baseline refuses to run without an identity decision:

```bash
python -m imu2text.sip baseline data/sip-19485897/UNO_B.xlsx \
  --groups subject-codes.csv --seed 0 \
  --output results/sip-local/UNO_B-grouped.json
```

`subject-codes.csv` must contain `row_index,subject_id` and exactly one mapping
for each original zero-based table row. Use verified pseudonymous codes, not
names or invented unique identifiers. Group splitting keeps each supplied
code on one side; the code cannot verify whether those codes reflect people.
Do not commit the mapping.

For a diagnostic smoke experiment only, explicitly opt into a row split:

```bash
python -m imu2text.sip baseline data/sip-19485897/UNO_B.xlsx \
  --exploratory-row-split --seed 0 \
  --output results/sip-local/UNO_B-row-experiment.json
```

The target is `Z <= -1.5`, a task-specific low-speed label, not a diagnosis.
Demographic fields and Z never enter the predictors. Median imputation and
scaling are fitted on training rows only. The fixed model uses balanced
logistic regression; no search selects a seed or hyperparameters against the
test results. Reports include versions, model settings, split fingerprint,
class counts, confusion matrices and positive-class metrics. Average precision
is reported by that name, not relabelled as the paper's PR-curve area.
Nonconvergence fails instead of producing a normal-looking report.

Exact feature duplicates crossing the split also fail. Do not delete this
check to obtain a score. Audits found two matching-feature rows each in UNO_B,
NUM_B and NUM_C, and four in UNO_C. They may be repeated records or reflect a
source-data issue; the files do not settle their identity. A verified grouping
or an approved exclusion protocol is needed before treating these as validation.

The [recorded smoke run](../results/sip/2026-11-18.json) uses seed 0 and the
explicit exploratory row protocol. UNO_B and NUM_C complete; UNO_C and NUM_B
stop at the cross-split duplicate check. Even the completed experiments have
unknown subject overlap and same-side duplicates. Their scores are software
smoke results, not evidence for screening, Vahini hardware or generalization.
Only aggregate reports are committed; no participant records, mappings,
individual predictions or fitted model files are exported.

### Verification

```bash
python -m pip install pytest
python -m pytest tests/test_sip.py tests/test_docs_style.py -q
# Optional real-file integration check after the pinned download:
SIP_DATA_DIR=data/sip-19485897 python -m pytest tests/test_sip.py -q
```

The environment-variable prefix is POSIX syntax; on PowerShell set
`$env:SIP_DATA_DIR = "data/sip-19485897"` before running pytest.
Synthetic tests cover schema drift, target exclusion, training-only transforms,
identity requirements, grouping, duplicate leakage, checksum failures,
missing targets, convergence failure and input-overwrite protection.
