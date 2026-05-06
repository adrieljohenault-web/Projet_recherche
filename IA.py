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
from formatage_verif import v1, v2, v3, vin3


X_train = entree
# y_train = np.array([sortie_fichier(i) for i in range(1,n_valeurs_calc+1)])
# y_trainl = y_train[:,0]
# C'est trop long autant le faire une fois et le stocker

X_val = np.array([[vin3[i][j] for j in range(1, 4)] for i in range(len(vin3))])
y_val1 = np.array([[v1[i][j] for j in range(1, 3)] for i in range(len(v1))]) # point de la sonde 1
y_val2 = np.array([[v2[i][j] for j in range(1, 3)] for i in range(len(v2))]) # point de la sonde 2
y_val3 = np.array([[v3[i][j] for j in range(1, 3)] for i in range(len(v3))]) # point de la sonde 3

# On va commencer avec un modèle de type Random_Forest, car il demandera peu voire pas de fine-tuning
# Pour s'aider éventuellement TP5 de 1ISDO