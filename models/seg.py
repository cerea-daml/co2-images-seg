# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
from tensorflow import keras
import segmentation_models as ext_sm
from dataclasses import dataclass, field
from functools import partial
import models.Unet_backboned as Uback


def Unet_4(input_shape, Nclasses):
    """Unet ~ 2millions parameters - four phases."""

    [Ny, Nx, Nchannels] = input_shape

    inputs = keras.layers.Input((Ny, Nx, Nchannels))

    c1 = keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(inputs)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = keras.layers.Dropout(0.1)(c2)
    c2 = keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = keras.layers.Dropout(0.2)(c3)
    c3 = keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = keras.layers.Dropout(0.2)(c4)
    c4 = keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = keras.layers.Conv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = keras.layers.Dropout(0.3)(c5)
    c5 = keras.layers.Conv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c5)

    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = keras.layers.Dropout(0.2)(c6)
    c6 = keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = keras.layers.Dropout(0.2)(c7)
    c7 = keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = keras.layers.Dropout(0.1)(c8)
    c8 = keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    c9 = keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = keras.layers.Dropout(0.1)(c9)
    c9 = keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = keras.layers.Conv2D(Nclasses, (1, 1), activation="sigmoid")(c9)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model


def Unet_5(input_shape, Nclasses, dropout_rate=0.1):
    """Unet ~ 5millions parameters - five phases."""

    conv2d_elu_he = partial(
        keras.layers.Conv2D,
        activation="elu",
        kernel_initializer="he_normal",
        padding="same",
    )

    [Ny, Nx, Nchannels] = input_shape
    N_chan_i = 16

    inputs = keras.layers.Input((Ny, Nx, Nchannels))

    """Downsampling with downsampler and convolutions."""
    d_c0 = conv2d_elu_he(N_chan_i, (3, 3))(inputs)
    d_c0 = keras.layers.Dropout(dropout_rate)(d_c0)
    d_c0 = conv2d_elu_he(N_chan_i, (3, 3))(d_c0)
    d_c0 = keras.layers.BatchNormalization()(d_c0)

    d_d0 = keras.layers.MaxPooling2D((2, 2))(d_c0)

    d_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(d_d0)
    d_c1 = keras.layers.Dropout(dropout_rate)(d_c1)
    d_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(d_c1)
    d_c1 = keras.layers.BatchNormalization()(d_c1)

    d_d1 = keras.layers.MaxPooling2D((2, 2))(d_c1)

    d_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(d_d1)
    d_c2 = keras.layers.Dropout(dropout_rate * 2)(d_c2)
    d_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(d_c2)
    d_c2 = keras.layers.BatchNormalization()(d_c2)

    d_d2 = keras.layers.MaxPooling2D((2, 2))(d_c2)

    d_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(d_d2)
    d_c3 = keras.layers.Dropout(dropout_rate * 2)(d_c3)
    d_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(d_c3)
    d_c3 = keras.layers.BatchNormalization()(d_c3)

    d_d3 = keras.layers.MaxPooling2D((2, 2))(d_c3)

    d_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(d_d3)
    d_c4 = keras.layers.Dropout(dropout_rate * 3)(d_c4)
    d_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(d_c4)
    d_c4 = keras.layers.BatchNormalization()(d_c4)

    d_d4 = keras.layers.MaxPooling2D((2, 2))(d_c4)

    """Mid-part."""
    m = conv2d_elu_he(N_chan_i * 16, (3, 3))(d_d4)
    m = keras.layers.Dropout(dropout_rate * 3)(m)
    m = conv2d_elu_he(N_chan_i * 16, (3, 3))(m)
    m = keras.layers.BatchNormalization()(m)

    """Upsampling with upsampler, residual, and convolutions."""
    u_u4 = keras.layers.Conv2DTranspose(
        N_chan_i * 16, (2, 2), strides=(2, 2), padding="same"
    )(m)
    u_r4 = keras.layers.concatenate([u_u4, d_c4])

    u_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(u_r4)
    u_c4 = keras.layers.Dropout(dropout_rate * 3)(u_c4)
    u_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(u_c4)
    u_c4 = keras.layers.BatchNormalization()(u_c4)

    u_u3 = keras.layers.Conv2DTranspose(
        N_chan_i * 8, (2, 2), strides=(2, 2), padding="same"
    )(u_c4)
    u_r3 = keras.layers.concatenate([u_u3, d_c3])

    u_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(u_r3)
    u_c3 = keras.layers.Dropout(dropout_rate * 2)(u_c3)
    u_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(u_c3)
    u_c3 = keras.layers.BatchNormalization()(u_c3)

    u_u2 = keras.layers.Conv2DTranspose(
        N_chan_i * 4, (2, 2), strides=(2, 2), padding="same"
    )(u_c3)
    u_r2 = keras.layers.concatenate([u_u2, d_c2])

    u_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(u_r2)
    u_c2 = keras.layers.Dropout(dropout_rate * 2)(u_c2)
    u_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(u_c2)
    u_c2 = keras.layers.BatchNormalization()(u_c2)

    u_u1 = keras.layers.Conv2DTranspose(
        N_chan_i * 2, (2, 2), strides=(2, 2), padding="same"
    )(u_c2)
    u_r1 = keras.layers.concatenate([u_u1, d_c1])

    u_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(u_r1)
    u_c1 = keras.layers.Dropout(dropout_rate)(u_c1)
    u_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(u_c1)
    u_c1 = keras.layers.BatchNormalization()(u_c1)

    u_u0 = keras.layers.Conv2DTranspose(
        N_chan_i, (2, 2), strides=(2, 2), padding="same"
    )(u_c1)
    u_r0 = keras.layers.concatenate([u_u0, d_c0])

    u_c0 = conv2d_elu_he(N_chan_i, (3, 3))(u_r0)
    u_c0 = keras.layers.Dropout(dropout_rate)(u_c0)
    u_c0 = conv2d_elu_he(N_chan_i, (3, 3))(u_c0)
    u_c0 = keras.layers.BatchNormalization()(u_c0)

    outputs = keras.layers.Conv2D(Nclasses, (1, 1), activation="sigmoid")(u_c0)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model


@dataclass
class Seg_model_builder:
    """Return appropriate segmentation model."""

    name: str = "Unet_efficientnetb0"
    input_shape: list = field(default_factory=lambda: [160, 160, 1])
    classes: int = 1
    dropout_rate: np.float32 = 0.2

    def get_model(self):
        """Return segmentation model, from local or segmentation_models (old)."""
        
        if self.name.startswith("Unet_efficientnetb"):
            encoder_name = {
                "Unet_efficientnetb0": "EfficientNetB0",
                "Unet_efficientnetb1": "EfficientNetB1",
                "Unet_efficientnetb2": "EfficientNetB2",
                "Unet_efficientnetb3": "EfficientNetB3",
                "Unet_efficientnetb4": "EfficientNetB4",
                "Unet_efficientnetb5": "EfficientNetB5",
                "Unet_efficientnetb6": "EfficientNetB6",                
            }[self.name]
            print (encoder_name, self.input_shape, self.classes)
            model = Uback.Unet(
                encoder_name,
                input_shape=self.input_shape,
                classes=self.classes,
                drop_encoder_rate=self.dropout_rate,
            )

        else:
            model_names = {"Unet_4": Unet_4, "Unet_5": Unet_5}
            model = model_names[self.name](self.input_shape, self.classes)

        return model

    """
    model = ext_sm.Unet(backbone_name="efficientnetb1", 
                                encoder_weights=None, input_shape=self.input_shape)
    """
