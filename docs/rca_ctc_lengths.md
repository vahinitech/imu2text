# CTC alignment and validation root-cause analysis

## Experiment question

Does correcting sequence alignment and validation improve recognition of unseen
writers on the downloaded right-handed OnHW-Words500 archive?

The planned comparison uses official writer-independent fold 0, seed 0, 15
epochs, batch size 32, two convolution blocks, and one 32-unit bidirectional
LSTM. The parent and corrected pipelines use the same official test recordings.
This smaller encoder is a CPU experiment configuration, not the CLI default.
The existing character-classification benchmark measures a different task.

The prior expectation is lower CER after fixing alignment. A CER increase of
two percentage points calls for investigation rather than an improvement claim.
Repeat the same deterministic configuration to measure run-to-run variation on
this archive. Compare the corrected pipeline with random validation as an
ablation of writer-disjoint validation. Select no configuration by test accuracy.

## Root causes

### Padding was treated as recorded motion

The parent `seq2seq.run` filled every CTC input length with `maxlen // 4`.
For an 800-frame tensor, every example therefore had 200 output frames, even
when its recording was much shorter. Greedy and lexicon decoding used the same
constant. The bidirectional LSTM also processed the entire padded tail.

CTC could align target characters to padding; the reverse recurrent pass could
incorporate padding into valid-frame features. The corrected pipeline passes
each recording's length after pooling to the loss, recurrent mask, and both
decoders. Regression tests inject predictions into the padded tail and verify
that they cannot add output characters.

### Long recordings lost their endings

Post-truncation discarded everything after frame 800 while retaining the complete
word label. In fold 0, 51 of the 19,915 nonempty training recordings exceed that
limit. The corrected preprocessing interpolates the entire recording into fixed
duration bounds, 80 to 800 frames. The endpoints remain present.

Those bounds depend on input duration, not on target text. Upsampling a short
recording provides alignment frames but adds no sensor information. Compressing
a very long recording changes its temporal frequencies and can lose detail.
The benchmark must establish the effect of this policy; interpolation is not a
guarantee of better recognition.

### The CTC feasibility check missed repeated characters

An adjacent repeated character needs an intervening blank. For example, `aa`
requires three output frames, whereas `ab` requires two. The parent checked only
the largest label length against the global padded length.

The new check requires, for each recording, at least the label length plus its
number of adjacent repeats. The downloaded fold contains two nonempty training
recordings and one nonempty test recording that cannot align after native
fourfold downsampling. Fixed duration resampling makes them feasible without
discarding them from evaluation or deriving input lengths from their labels.

### Validation did not represent unseen writers

The official Words500 test partition is writer-disjoint, but the parent carved
validation samples randomly from the training partition. Writers could appear
in both fitting and validation, making checkpoint selection answer an easier
question than final evaluation.

The corrected Words500 entry point passes the available pseudonymous writer
codes into a group split. Validation holds out training writers; the official
test partition is unchanged. A check rejects overlapping test writers when the
writer-independent protocol is requested. The ablation retains random
validation to assess the contribution of this change separately.

### Reaching the epoch limit could bypass checkpoint restoration

The installed Keras 2.15 implementation restores best weights inside the early
stopping branch. Reaching the epoch limit before patience expires leaves the
last epoch's weights in place. `RestoreBest` restores the recorded validation
checkpoint in `on_train_end` as well. This applies to character and sequence
models. A regression test finishes training without triggering early stopping
and checks that the better earlier weights are restored.

### Character-model selection consulted test accuracy

The character benchmark sorted candidates by `test_acc` and exported the first
candidate's predictions. It now ranks by `val_acc`, with stable ordering for
ties. Test scores describe the selected candidate; they do not select it.
The comparison-table runner accepts the updated result line.

## Methods introduced

The recognizer remains CNN + BiLSTM + CTC. These changes add per-recording
alignment lengths, explicit recurrent masking, fixed-bound linear resampling,
grouped validation, and checkpoint restoration at the epoch limit. They do not
introduce a transformer or a new recognition architecture. The lexicon prefix
beam decoder already existed; it now receives the same valid lengths as greedy
decoding. Its dictionary comes only from fitting labels.

The output alphabet comes from the published Words500 vocabulary, or from
training labels for a custom dataset. It is no longer inferred from test labels.
Normalization is still fitted on training only. Incremental scaler fitting and
direct writes into the padded tensor reduce temporary array copies.
Training copies one minibatch at a time rather than converting entire padded
partitions into additional TensorFlow arrays. Each batch trims padding beyond
its longest recording; the model accepts a variable time dimension. Its shuffle
uses a seeded NumPy
generator. This changes minibatch order relative to the parent's Keras array
adapter, so the aggregate comparison cannot assign every point to masking alone.

## Dataset findings and reporting boundaries

The loader drops three empty training recordings and eight empty official test
recordings. Evaluation therefore contains 5,292 recordings rather than the
archive's 5,300. No additional test recordings are removed by these fixes.
The downloaded fold contains 501 distinct decoded strings across both halves;
there are no empty labels or leading/trailing-whitespace variants. The code
preserves those strings rather than forcing a 500-entry vocabulary.

CER and WER are edit rates, where lower is better. Exact word accuracy is
reported separately. The old character accuracy is not comparable to word
accuracy or to one minus CER. Single-fold, single-seed measurements do not
establish a five-fold mean or performance on Vahini's own pen hardware.

## References

- [TensorFlow CTC loss and input lengths](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss).
- [Keras LSTM masking](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM).
- [Keras 2.15 callback implementation](https://github.com/keras-team/keras/blob/v2.15.0/keras/src/callbacks.py).
- [Fraunhofer OnHW dataset](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html).

## Measurements

The benchmark results and measured attribution will be recorded here after the
preselected runs finish, before the pull request is created.
