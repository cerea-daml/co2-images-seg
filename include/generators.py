# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
from treeconfigparser import TreeConfigParser
from tensorflow import keras
from dataclasses import dataclass


@dataclass
class Generator:
    """Generator used to create augmented images - and labels - for DL training."""
    model_purpose: str
    batch_size: int = 32
    rotation_range: int = 0
    shift_range: float = 0
    flip: bool = False
    shear_range: float = 0
    zoom_range: float = 0

    def __post_init__(self):
        self.createDataGenerator()

    def createDataGenerator(self):
        """Create data generator."""

        data_gen_args = dict(
            rotation_range=self.rotation_range,
            width_shift_range=self.shift_range,
            height_shift_range=self.shift_range,
            horizontal_flip=self.flip,
            vertical_flip=self.flip,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
        )

        self.image_datagen = keras.preprocessing.image.ImageDataGenerator(
            **data_gen_args
        )

        if self.model_purpose.startswith("segmentation"):
            self.mask_datagen = keras.preprocessing.image.ImageDataGenerator(
                **data_gen_args
            )

    def flow(self, x_data, y_data):
        """Flow on x (img) and y (label) data to generate:
        - segmentation: augmented images and augmented corresponding labels
        - regression: augmented images and non-augmented corresponding labels 
        (emissions rate kept unchanged).
        """

        seed = 27

        if self.model_purpose.startswith("segmentation"):
            self.image_generator = self.image_datagen.flow(
                x_data, seed=seed, batch_size=self.batch_size, shuffle=False
            )
            self.mask_generator = self.mask_datagen.flow(
                y_data, seed=seed, batch_size=self.batch_size, shuffle=False
            )

            self.train_generator = zip(self.image_generator, self.mask_generator)

        elif self.model_purpose == "inversion":
            self.train_generator = self.image_datagen.flow(
                x_data, y_data, seed=seed, batch_size=self.batch_size, shuffle=False
            )

        else:
            print("Unknown model purpose in Generator")

        return self.train_generator

    def next(self):
        return self.image_generator.next(), self.mask_generator.next()


