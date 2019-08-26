#!/usr/bin/python3

from __future__ import division
from itertools import izip,count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt,ones,convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
style.use('fivethirtyeight')


data_as_frame = pd.read_csv('sunspots.txt',sep='\t');
print(data_as_frame.head())

