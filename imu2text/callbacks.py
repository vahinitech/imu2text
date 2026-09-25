"""Validation-based checkpoint restoration across Keras stopping paths."""

from tensorflow.keras.callbacks import EarlyStopping


class RestoreBest(EarlyStopping):
    """Restore the best validation weights even when the epoch budget ends first.

    Keras 2.15 restores in its early-stop branch only. A completed fixed-length
    benchmark must evaluate the same validation-selected checkpoint policy.
    """

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
