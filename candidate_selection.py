import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import os
import json
import joblib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM

import lightfm
lightfm.__version__

model = joblib.load(open('models/lightfm_model.pkl', 'rb'))
model_text = joblib.load(open('models/lightfm_model_text.pkl', 'rb'))
user_features = joblib.load(open('models/user_features.pkl', 'rb'))