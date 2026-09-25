"""Checkpoint restoration must also work when training exhausts its epoch budget."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

# pylint: disable=wrong-import-position
from imu2text.callbacks import RestoreBest


def test_best_weights_restored_without_triggering_early_stop():
    class Model:
        stop_training = False
        weights = [np.array([1.0])]

        def get_weights(self):
            return self.weights

        def set_weights(self, weights):
            self.weights = weights

    model = Model()
    callback = RestoreBest(monitor="val_loss", patience=8, restore_best_weights=True)
    callback.set_model(model)
    callback.on_train_begin()
    callback.on_epoch_end(0, {"val_loss": 1.0})
    model.weights = [np.array([2.0])]
    callback.on_epoch_end(1, {"val_loss": 2.0})
    assert not model.stop_training
    callback.on_train_end()
    np.testing.assert_array_equal(model.weights[0], [1.0])
