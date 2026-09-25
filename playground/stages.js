// The pipeline stages the playground shows. Edit this file to add a method.
//
// Every accuracy here must come from a committed file or doc in this repo,
// named in `source`, with its split in `conditions`. See AGENTS.md.
window.PLAYGROUND_STAGES = {
  repo: "https://github.com/vahinitech/imu2text",

  filters: [
    {
      id: "none",
      name: "No filter",
      code: "imu2text/filters.py",
      accuracy: 67.32,
      conditions: "CNN+BiLSTM, 30 epochs, seed 0, official both/indep/fold0, 52 classes",
      source: "docs/rca_filters.md",
      note: "The reference run the two filters are compared with.",
    },
    {
      id: "lowpass",
      name: "Low-pass, 15 Hz",
      code: "imu2text/filters.py",
      accuracy: 66.39,
      conditions: "CNN+BiLSTM, 30 epochs, seed 0, official both/indep/fold0, 52 classes",
      source: "docs/rca_filters.md",
      note: "Cost 0.93 points. More than half the accelerometer energy is above 15 Hz, and the model uses it.",
    },
    {
      id: "orientation",
      name: "Orientation (Madgwick)",
      code: "imu2text/filters.py",
      accuracy: 63.76,
      conditions: "CNN+BiLSTM, 30 epochs, seed 0, official both/indep/fold0, 52 classes",
      source: "docs/rca_filters.md",
      note: "Measured before the gyroscope-unit fix, so this number does not measure the filter. Re-running it is an open task.",
    },
  ],

  models: [
    {
      id: "cnn_bilstm_attn",
      name: "CNN + BiLSTM with attention",
      code: "imu2text/models.py",
      paper: "Ott et al., ACM MM 2022 (CNN+BiLSTM); attention pooling added in this repo",
    },
  ],

  // Methods the page has an empty card for. Link an issue when there is one.
  open: [
    {
      name: "Kalman filter",
      stage: "filter",
      issue: null,
      why: "Not built. A filter that fuses the two accelerometers and the gyroscope before the model sees them.",
    },
    {
      name: "Hybrid classical + deep model",
      stage: "model",
      issue: 12,
      why: "Hand-made features next to the network. Filed with a prediction of 0 to +2 points.",
    },
    {
      name: "Separate letter and case",
      stage: "model",
      issue: 10,
      why: "Most errors are a letter read as its other case. A 26-way head plus a case head targets that directly.",
    },
    {
      name: "Case from word context",
      stage: "model",
      issue: 11,
      why: "In words, position decides case. The lexicon decoder already exists.",
    },
    {
      name: "Know when to abstain",
      stage: "result",
      issue: 13,
      why: "Report accuracy on the letters the model is sure about, and flag the rest.",
    },
  ],
};
