# co2-images-seg
The official code for "Segmentation of XCO2 images with deep learning: application to synthetic plumes from cities and power plants", submitted to " Geoscientific Model Development".

[![DOI](https://zenodo.org/badge/570560732.svg)](https://zenodo.org/badge/latestdoi/570560732)

## Description

In this work, we show a proof-of-concept that deep learning can segment XCO2 images 
to recover anthropogenic hotspot plumes from XCO2 images only.

The scripts and module are written in python and we use the deep learning interface Tensorflow.

To use these scripts, the datasets of fields and plumes have to be downloaded from [seg-zenodo](https://zenodo.org/record/7362580).
Weights of already trained models are provided at [seg-zenodo](https://zenodo.org/record/7362580).
A configuration file example is provided in the folder examples.
The two lines of the configuration files have to been modified:
- `data.directory.main` has to be assigned to the directory where the netcdf datasets have been stored.
- `data.directory.name` has to be assigned to the name of a netcdf dataset.

The netcdf datasets can also be generated directly from the [SMARTCARB](https://zenodo.org/record/4034266#.Yt6btp5BzmE) dataset.
The data generation scripts are not included in this repository, they are available on request.


After data collection/generation, the neural networks can be trained with the python `main.py` script (see `examples/train.ipynb`).
This script trains the Unet neural network as described in the manuscript. 

After training the neural networks, they can be used in the `examples/test.ipynb` notebook to generate predictions on the chosen test dataset.

If you have further questions, please feel free to contact us or to create a GitHub issue.


##  Authors and acknowledgment

This project has been funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement 958927 (Prototype system for a Copernicus CO2 service)
CEREA is a member of Institut Pierre Simon Laplace (IPSL).

##  Support

Contact: joffrey.dumont@enpc.fr
