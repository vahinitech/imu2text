"""IMU handwriting recognition: OnHW dataset loaders and benchmark models.

Modules
-------
``chars`` / ``symbols`` / ``words``
    Loaders for the published Fraunhofer OnHW archives. Each was verified
    against the real ZIPs rather than a fixture, because the two disagreed
    in four separate ways.
``download``
    Catalog of the OnHW archives and a safe extractor.
``augment``
    IMU-specific augmentation transforms and the two named policies.
``models``
    The honest benchmark suite: baselines, CNN+BiLSTM, the attention-pooled
    variant, and both writer-independent and official-split evaluation.
``seq2seq``
    CTC sequence-to-sequence recognition for words and equations.

Nothing here imports TensorFlow at module scope, so the loaders stay usable
on a machine that has only numpy. ``models`` and ``seq2seq`` do need it.
"""

__all__ = ["augment", "chars", "download", "models", "seq2seq", "symbols", "words"]
