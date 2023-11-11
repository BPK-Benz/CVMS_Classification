import os,glob
import pandas as pd
import numpy as np
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


generalresult = []

for modelfile in glob.glob('BestModel/*'):
    bestmodel = modelfile.split('/')[1].split('_6')[0]
    for _ in glob.glob('Prepare_Stage2/*.csv'):
        if bestmodel in _.split('Net_')[1].split('s_')[0]:
            loaded_model = load(modelfile)
            X_test = pd.read_csv(_).iloc[:,:-1]
            Y_test = pd.read_csv(_).iloc[:,-1]
            result = loaded_model.score(X_test, Y_test)
            dataName = 'Data_'+_.split('Net_')[1].split('s_')[1].split('_6')[0]
            generalresult += [['Model_'+bestmodel,dataName, round(result,3)]]

wideDf = pd.DataFrame(generalresult, columns=['Model','New_Dataset','Accuracy'])
pd.pivot_table(wideDf, columns='Model', index='New_Dataset', values='Accuracy').to_csv('ModelGeneralization.csv')