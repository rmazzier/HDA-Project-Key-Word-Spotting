# Key-Word Spotting using Deep Neural Networks

Project repository for the Human Data Analytics course exam, Data Science Master Degree, Padova AA 2020-2021.

## Project Overview

The Keyword Spotting (KWS) task consists in the detection of a certain predetermined set of keywords from a stream of user utterances. Deep learning models have proved to give highly accurate results, while remaining lightweight and suitable for running in mobile devices.
In this project, a variety deep neural architectures for KWS are tested. Specifically, I focused on architectures based on the attention mechanism. All models are trained on the [Google Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands?hl=en), on the 12kws task and the 35kws task.

A detailed report of all the work can be found in the project report, which is placed in `project_report/project_report.pdf` .

## Code Details

In order to run the code, the dataset folder (downloadable [here](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)) must be placed inside the `data` folder, and must have name `speech_commands_v0.02`.

All proposed models are defined in `models.py`. To train all the models, run the `train_models.py` python script. All the hyperparameters are defined in `hyperparameters.py`.

Two Jupyter Notebooks are provided:
 - `Input Pipeline.ipynb` [![NBViewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/rmazzier/HDA-Project-Key-Word-Spotting/blob/main/Input%20Pipeline.ipynb):
 contains a demonstration of how the input pipeline for the project works. 
 - `Models Evaluation.ipynb` [![NBViewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/rmazzier/HDA-Project-Key-Word-Spotting/blob/main/Models%20Evaluation.ipynb): here, all the weights from the trained models are loaded, in order to evaluate them. At the end, all the code to produce the plots present in the project report is provided. To run this notebook, the `models` folder must contan the weights of the trained models. To get them, one must train all the models by executing the `train_models.py` file. Alternatively, they can be found [here](https://drive.google.com/file/d/1c74-zhuSnt1hY_qqpew3TAvMFKTK3VdD/view?usp=sharing).

