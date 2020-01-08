# Import statements, helper functions and variables normalization factors
# Created by: Arif Bayirli


from __future__ import print_function
import numpy as np
import uproot
import time
import sys
import glob
import math
import os
import pickle
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import mlflow
from tensorboardX import SummaryWriter
import tempfile
from configurations import *


def plot_roc_curve(fpr, tpr, label = None, color = "C0", line='-'):
    """Plots the ROC curve given True Positive Rate (tpr) and 
    False Positive Rate (fpr)
    """
    plt.plot(tpr, fpr, linewidth = 2, label= label, color = color, linestyle=line)
    plt.axis([0,1,0,1])
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")

    
def findFiles(path): 
    """Finds all of the file in a directory whose path is given
    """
    return glob.glob(path)

def timeSince(since):
    """calculates time since in human readable form
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def file_dictionary_maker(file_paths, data_year):
    """creates a dictionary of all root files in the training and testing directories
    given in the configuration file
    """
    mc_e_train_files = reduce(lambda x,y: x+y,[findFiles(dir + '/*.root') for dir in file_paths["mc_e_train_dirs"]])
    mc_e_test_files = reduce(lambda x,y: x+y,[findFiles(dir + '/*.root') for dir in file_paths["mc_e_train_dirs"]])
    mc_m_train_files = reduce(lambda x,y: x+y,[findFiles(dir + '/*.root') for dir in file_paths["mc_m_train_dirs"]])
    mc_m_test_files = reduce(lambda x,y: x+y,[findFiles(dir + '/*.root') for dir in file_paths["mc_m_test_dirs"] ])

    data_test_files = reduce(lambda x,y: x+y,[findFiles(dir + '/*.root') for dir in file_paths["data_test_dirs"]])

    filelist_train = {"electron_mc": mc_e_train_files, "muon_mc": mc_m_train_files}
    filelist_test = {"electron_mc": mc_e_test_files, "muon_mc": mc_m_test_files, f"electron_probes{data_year}": data_test_files, f"muons{data_year}": data_test_files}

    return filelist_train, filelist_test