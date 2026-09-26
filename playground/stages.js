// The pipeline stages the playground shows. Edit this file to add a method.
//
// Every accuracy here must come from a committed file or doc in this repo,
// named in `source`, with its split in `conditions`. See AGENTS.md.
window.PLAYGROUND_STAGES = {
  repo: "https://github.com/vahinitech/imu2text",

  filters: [
    {
      id: "none",
      name: "Raw signal",
      code: "imu2text/filters.py",
      accuracy: 67.32,
      conditions: "CNN+BiLSTM, 30 epochs, seed 0, official both/indep/fold0, 52 classes",
      source: "docs/rca_filters.md",
      note: "Nothing removed: the model sees exactly what the pen recorded.",
    },
    {
      id: "lowpass",
      name: "Smoothed",
      code: "imu2text/filters.py",
      accuracy: 66.39,
      conditions: "CNN+BiLSTM, 30 epochs, seed 0, official both/indep/fold0, 52 classes",
      source: "docs/rca_filters.md",
      note: "Removes fast wiggles above 15 times a second. It made the model slightly worse (0.93 points): some of those fast wiggles are real handwriting.",
    },
    {
      id: "orientation",
      name: "Tilt removed",
      code: "imu2text/filters.py",
      accuracy: 63.76,
      conditions: "CNN+BiLSTM, 30 epochs, seed 0, official both/indep/fold0, 52 classes",
      source: "docs/rca_filters.md",
      note: "Works out how the pen is tilted and removes it. This number was measured before a bug in the filter was fixed, so it says little; measuring it again is an open task.",
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
      name: "Pen-tip trajectories",
      stage: "model",
      issue: null,
      why: "OnHW-wordsTraj pairs IMU recordings with the tip's path from a tablet: 16,752 samples from 2 writers. No loader or model yet, and with 2 writers every result is writer-dependent.",
    },
    {
      name: "Adapting to a new writer",
      stage: "model",
      issue: null,
      why: "Domain adaptation (for example CORAL) adjusts the model to a writer's own recordings. It changes what a split means, so a result has to say what the model saw.",
    },
    {
      name: "Know when to abstain",
      stage: "result",
      issue: 13,
      why: "Report accuracy on the letters the model is sure about, and flag the rest.",
    },
  ],
};
