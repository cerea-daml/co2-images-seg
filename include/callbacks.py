# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np
from treeconfigparser import TreeConfigParser
from tensorflow import keras
from wandb.keras import WandbCallback
import wandb


def initiate_wb(config: TreeConfigParser) -> None:
    """Initiate Weight and Biases."""
                
    if config.get("model.callback.wandb"):
        config_wb = {
            "data": config.get("data.directory.name"),
            "model": config.get("model.name"),
            "rot": config.get_float("data.input.aug.rot.range"),
            "flip": config.get_bool("data.input.aug.flip.bool"),
            "shear": config.get_float("data.input.aug.shear.range"),
            "zoom": config.get_float("data.input.aug.zoom.range"),
            "dropout_rate": config.get("model.dropout_rate"),
        }        
        
        if config.get("data.output.label.choice") == "segmentation":
            config_wb["shift"] = config.get_float("data.input.aug.shift.range")
            config_wb["w_min"] = config.get_float("data.output.label.weight.min")
            config_wb["w_max"] = config.get_float("data.output.label.weight.max")
            config_wb["w_curve"] = config.get("data.output.label.weight.curve")
        
        if config.get("data.output.label.choice") == "inversion":
            supp_inputs = config.get_stringlist("data.input.supps.list")
            if len(supp_inputs) == 0:
                add_1 = None
                add_2 = None
            elif len(supp_inputs) == 1:
                add_1 = supp_inputs[0]
                add_2 = None
            elif len(supp_inputs) == 2:
                add_1 = supp_inputs[0]
                add_2 = supp_inputs[1]

            config_wb["add_1"] = add_1
            config_wb["add_2"] = add_2
            config_wb["loss"] = config.get("model.loss")
            
        wandb.init(project=config.get("data.output.label.choice"), 
                   config=config_wb, 
                   name=config.get("orga.save.folder"))


def create_list_callbacks(
    save_dir: str,
    save_folder: str,
    modelcheckpoint: bool = False,
    reducelronplateau: bool = False,
    earlystopping: bool = False,
    wandb: bool = False,
) -> list:
    """Create a list of callbacks used during the training phase."""

    dir_save_model = os.path.join(
        save_dir, save_folder
    )

    list_callbacks = list()

    if modelcheckpoint:
        modelcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(dir_save_model, "weights_cp_best.h5"),
            save_weights_only=False,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            verbose=1,
        )
        list_callbacks.append(modelcheckpoint_cb)

    if reducelronplateau:
        reducelronplateau_cb = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            verbose=0,
            min_delta=5e-3,
            cooldown=0,
            min_lr=5e-5,
        )
        list_callbacks.append(reducelronplateau_cb)

    if earlystopping:
        earlystopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=5e-4,
            patience=50,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        list_callbacks.append(earlystopping_cb)

    if wandb:
        list_callbacks.append(WandbCallback())

    return list_callbacks
