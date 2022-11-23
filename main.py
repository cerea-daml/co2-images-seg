# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------


import bash
from model_training import Model_training_manager

if __name__ == "__main__":

    in_command = bash.config_file_names_from_command()
    config_file = in_command[0]

    model_trainer = Model_training_manager(config_file)
    model_trainer.run()
    model_trainer.save()
