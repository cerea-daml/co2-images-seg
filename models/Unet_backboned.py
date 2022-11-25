#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# heavily inspired from segmentation_models https://github.com/qubvel/segmentation_models
#----------------------------------------------------------------------------

import numpy as np
from tensorflow import keras
from dataclasses import dataclass
from functools import partial

# ---------------------------------------------------------------------
# U-net decoder blocks
# ---------------------------------------------------------------------

def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""
    def wrapper(input_tensor):

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )(input_tensor)

        if use_batchnorm:
            x = keras.layers.BatchNormalization()(x)

        if activation:
            x = keras.layers.Activation(activation)(x)

        return x

    return wrapper

def Conv3x3BnReLU(filters, use_batchnorm, name=None):

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    def wrapper(input_tensor, skip=None):
        x = keras.layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = keras.layers.Concatenate(axis=3, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)
        return x

    return wrapper

# ---------------------------------------------------------------------
#  Unet Build func
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(160, 80, 40, 20, 10),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    x = keras.layers.Lambda(lambda x: x)(x)

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)


    x = keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = keras.layers.Activation(activation, name=activation)(x)

    model = keras.models.Model(input_, x)

    return model

def Unet(
        encoder_name="EfficientNetB0",
        input_shape=(160, 160, 1),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights=None,
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        drop_encoder_rate=0.2,
        **kwargs
):


    model_to_call = getattr(keras.applications, encoder_name)
    base_model = model_to_call(
        include_top=False,
        weights=encoder_weights,
        input_shape=input_shape,
        drop_connect_rate=drop_encoder_rate,
    )
    base_model.trainable = True        

    encoder_features = ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation')

    model = build_unet(
        backbone=base_model,
        decoder_block=DecoderUpsamplingX2Block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )
    
    return model
