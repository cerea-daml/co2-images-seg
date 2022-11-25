# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from treeconfigparser import TreeConfigParser
import wandb

try:
    import models.reg as rm
except ImportError:
    pass

import models.seg as sm
from data.Data import Data
import include.callbacks as callbacks
import include.generators as generators
import include.loss as loss
import include.optimisers as optimisers
from saver import Saver


def build_model(
    model_purpose: str,
    name: str,
    init_w: str,
    input_shape: np.ndarray,
    classes: int,
    dropout_rate: np.float32,
)  -> keras.Model:
    """Build SEG, INV, or PCR model."""

    if model_purpose.startswith("segmentation"):
        seg_builder = sm.Seg_model_builder(name, input_shape, classes, dropout_rate)
        model = seg_builder.get_model()

    elif model_purpose == "inversion":
        reg_builder = rm.Reg_model_builder(name, input_shape, classes, init_w)
        model = reg_builder.get_model()

    elif model_purpose == "pixel_concentration_retrieval":
        model = pwrm.Unet_2(input_shape, classes)

    return model


@dataclass
class Trainer:
    """Train CNN models."""

    generator: generators.Generator
    callbacks: list
    batch_size: int
    N_epochs: int

    def train_model(self, model, data) -> keras.Model:
        """Train model and evaluate validation."""
        self.history = model.fit(
            self.generator.flow(data.x.train, data.y.train),
            epochs=self.N_epochs,
            validation_data=(data.x.valid, data.y.valid),
            verbose=1,
            steps_per_epoch=int(np.floor(data.x.train.shape[0] / self.batch_size)),
            callbacks=self.callbacks,
            shuffle=True,
        )

        return model


class Model_training_manager:
    """Train CNN models for presence, segmentation, inversion, pixel concentration retrieval tasks."""

    def __init__(self, config_file: str) -> None:
        config = TreeConfigParser()
        config.readfiles(config_file)

        self.prepare_data(config)

        self.build_model(config)

        self.prepare_training(config)

        self.saver = Saver(config, config_file)

    def prepare_data(self, config: TreeConfigParser) -> None:
        """Prepare Data inputs to the neural network and outputs (=labels, targets)."""
        self.data = Data(config)
        self.data.prepare_input()
        self.data.prepare_output()

    def build_model(self, config: TreeConfigParser) -> None:
        """Build model."""
        self.model = build_model(
            config.get("data.output.label.choice"),
            config.get("model.name"),
            config.get("model.init"),
            self.data.x.fields_input_shape,
            self.data.y.classes,
            config.get_float("model.dropout_rate"),
        )
        self.model.compile(
            optimizer=optimisers.define_optimiser(config),
            loss=loss.define_loss(config.get("model.loss")),
            metrics=loss.define_metrics(config),
        )

    def prepare_training(self, config: TreeConfigParser) -> None:
        """Prepare the training phase."""
        callbacks.initiate_wb(config)
        generator = generators.Generator(
            config.get("data.output.label.choice"),
            config.get_int("model.batch_size"),
            config.get_int("data.input.aug.rot.range"),
            config.get_float("data.input.aug.shift.range"),
            config.get_bool("data.input.aug.flip.bool"),
            config.get_float("data.input.aug.shear.range"),
            config.get_float("data.input.aug.zoom.range"),
        )
        list_callbacks = callbacks.create_list_callbacks(
            config.get("orga.save.directory"),
            config.get("orga.save.folder"),
            config.get_bool("model.callback.modelcheckpoint"),
            config.get_bool("model.callback.reducelronplateau"),
            config.get_bool("model.callback.earlystopping"),
            config.get_bool("model.callback.wandb"),
        )
        self.trainer = Trainer(
            generator,
            list_callbacks,
            config.get_int("model.batch_size"),
            config.get_int("model.epochs.number"),
        )

    def run(self) -> None:
        """Train the model with the training data."""
        self.model = self.trainer.train_model(self.model, self.data)

    def save(self) -> None:
        """Save results of the run."""
        self.saver.save_model_and_weights(self.model)
        self.saver.save_data_shuffle_indices(
                self.data.eval_shuffler.ds_inds
        )
        self.saver.save_input_scaler(self.data.x.scaler)


