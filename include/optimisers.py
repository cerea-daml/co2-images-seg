# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
import os
import sys
from treeconfigparser import TreeConfigParser
import tensorflow_addons as tfa
from tensorflow import keras

# __________________________________________________________
# define_optimiser
def define_optimiser(config):

    if config.get("model.lr.decay.type") == "None":
        lr = config.get_float("model.lr.value")

    elif config.get("model.lr.decay.type") == "polynomial":
        lr_max = config.get_float("model.lr.max")
        lr_min = config.get_float("model.lr.min")
        decay_steps = config.get_int("model.lr.decay.steps")
        decay_power = config.get_int("model.lr.decay.power")
        lr = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr_max,
            decay_steps=decay_steps,
            end_learning_rate=lr_min,
            power=decay_power,
        )

    else:
        print("Learning rate of optimiser not defined")
        sys.exit()

    dicOpt = {
        "adam": keras.optimizers.Adam(learning_rate=lr),
        "yogi": tfa.optimizers.Yogi(learning_rate=lr),
    }

    optimiser_name = config.get("model.optimiser")
    opt = dicOpt[optimiser_name]

    return opt


# __________________________________________________________
