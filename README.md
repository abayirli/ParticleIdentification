# RNN Model for Particle Identification in ATLAS Experiment

GPU/CPU code for LSTM model training in Pytorch (version > 1.0.1) for the TRT Particle Identification qualification task **Particle Identification with Machine Learning in Transition Radiation Tracker (TRT) at ATLAS** - [ATL-COM-INDET-2020-001](https://cds.cern.ch/record/2705706).

It uses ntuples created by [TRTNTupler](https://gitlab.cern.ch/ddavis/TRTNTupler) (by Douglas Raymond Davis) and trains an RNN Model with the hit + track information.

The import statements, helper functions are included in the `helper_functions.py`. Model definition, track generator(dataloader) and model initialisation is done in `RNN_model_initialazer.py` file.

The code requires [uproot](https://github.com/scikit-hep/uproot) package from [scikit-hep](https://github.com/scikit-hep) library for reading ROOT files; [mlflow](https://mlflow.org/) and [tensorboardX](https://github.com/lanpa/tensorboardX) to log and observe the test statistics (e.g: train/test/validation losses) for each experiment; [scikit-learn](https://scikit-learn.org/) for some helper functions.

The model is based on [PyTorch](https://pytorch.org/) framework and it requires PyTorchversion greater than 1.0.1. It supports GPU training (given that PyTorch is installed with Cuda support). The script automatically detects the GPU and maps it into `device` parameters. If it is run on CPU, `device` is automatically set to 'cpu'. There is no modification needed.

The training and test (absolute) file paths needs to be modified in the main function (`configurations.py`) (i.e mc_e_train_dirs, mc_e_test_dirs etc...). Before you run the script, go over the parameters in the configuration fule listed change the relevant ones as you wish.

In order to run the training and testing, just run:

`python3 train_test_run.py`

The model is trained for a given epoch and it tests the resulting model on MC and Data test set in the end and plots the ROC curves. It also stores the PyTorch model file.

There are couple of Jupyter Notebooks in the **Notebooks** directory where you can run in Binder and reproduce the workflow partially (in order to reproduce everything, you will need full data set). In order to lanch the development enviroment, just click on the binder icon at the top.



Arif Bayirli
(TRT SW Group)
