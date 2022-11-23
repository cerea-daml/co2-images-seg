# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

from treeconfigparser import TreeConfigParser
import os
import numpy as np
import shutil
import pickle
import joblib


class Saver:
    """Saver of all results relevant to CNN model training experience."""

    def __init__(self, config, cfg):
        """Prepare directory to store results of the experiments."""

        self.dir_save_model = os.path.join(
            config.get("orga.save.directory"), config.get("orga.save.folder")
        )
        if os.path.exists(self.dir_save_model):
            shutil.rmtree(self.dir_save_model)
        os.makedirs(self.dir_save_model)
        self.save_cfg(cfg)

    def save_cfg(self, cfg):
        """Save config file."""
        shutil.copyfile(cfg, os.path.join(self.dir_save_model, "config.cfg"))

    def save_model_and_weights(self, model):
        """Save model and weights using keras built_in functions."""
        model.save(os.path.join(self.dir_save_model, "weights_model.h5"))

    def save_metrics(self, history, test_metrics=None):
        """Save train, valid, and test accuracy and loss metrics."""
        scores = ["accuracy", "val_accuracy", "loss", "val_loss"]
        for score in scores:
            if score in history.history:
                np.array(history.history[score]).tofile(
                    os.path.join(self.dir_save_model, score + ".bin")
                )
        if test_metrics:
            np.array(test_metrics).tofile(
                os.path.join(self.dir_save_model, "test_metrics.bin")
            )

    def save_training_time(self, training_time):
        """Save training time of the model."""
        np.array(training_time).astype("float").tofile(
            os.path.join(self.dir_save_model, "training_time.bin")
        )

    def save_data_shuffle_indices(self, ds_indices: dict):
        """Save shuffle indices."""
        with open(os.path.join(self.dir_save_model, 'tv_inds.pkl'), 'wb') as f:
            pickle.dump(ds_indices, f)
    
    def save_input_scaler(self, scaler):
        """Save input data sklearn scaler."""
        joblib.dump(scaler, os.path.join(self.dir_save_model, "scaler.save"))
