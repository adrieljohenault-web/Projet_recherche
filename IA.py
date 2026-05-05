import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from variables_globales import *
from Fonction_de_transfert import entree, sortie_fichier
from formatage_verif import vin3


X_train = entree
# y_train = np.array([sortie_fichier(i) for i in range(1,n_valeurs_calc+1)])
# y_trainl = y_train[:,0]
# C'est trop long autant le faire une fois et le stocker

X_val = vin3
y_val 

