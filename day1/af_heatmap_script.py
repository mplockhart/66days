# This script provides a way of outputting a heatmap of errors for AlphaFold models
# 1.  Importing all of the appropriate modules for data 
#     interpretation and plotting
# 2.  Obtaining the names of all of the pickle files and the 
#     directory of the outputs
# 3a. Stripping the predicted alignment error, pTM and ipTM from 
#     the pickle files for each model.
# 3b. Using the stripped data to plot the predicted alignement 
#     error while outputting the other parameters

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
import os
import seaborn as sns

files = glob.glob('result*.pkl', recursive=True)
dir_name = os.getcwd().split('/')[-1]

for file in files:
    df = pd.read_pickle(file) # add different models
    plt.figure(figsize=([10,8]))
    sns.heatmap(df['predicted_aligned_error'],cmap='viridis')
    ptm = np.round(df['ptm'],2)
    # Need to add a boolean operator to section iptm for multimers.
    iptm = str(np.round(df['iptm'],2))
    name = []
    file.split('_')
    name.append(file.split('_')[1])
    name.append(file.split('_')[2])
    model_name = ' '.join(name)
    file_model_name = '_'.join(name)
    plt.title(f"protein: {dir_name}, {model_name}, pTM: {ptm}, ipTM: {iptm}")
    plt.savefig(f'{dir_name}_{file_model_name}.png')
