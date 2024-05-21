# UTILS for the Danish HCA SA study

import os

import json
import sklearn
import pandas as pd
from importlib import reload
import numpy as np

#nltk.download('punkt')


# plot
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



# for calculating the measures
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()


# stats
from scipy.stats import mannwhitneyu
from scipy import stats
from scipy.stats import norm
from scipy.stats import spearmanr


