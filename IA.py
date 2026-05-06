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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

from variables_globales import *
from Fonction_de_transfert import entree, sortie_fichier
from formatage_verif import v1, v2, v3, vin3


X_train = entree

def load_from_npz() -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les données consolidées.
    Retourne (SH, SL) de shape (n_valeurs_calc, n_sortie, 8).

    Exemple d'usage dans IA.py :
        from consolidate_data import load_from_npz
        SH, SL = load_from_npz()
        y_trainh = SH   # (n_valeurs_calc, n_sortie, 8)
        y_trainl = SL
    """
    npz_path = os.path.join(path, "training_data.npz")
    data = np.load(npz_path)
    return data["SH"], data["SL"]

ytrainh,ytrainl = load_from_npz()

y_trainh = ytrainh[:,:,:2]
y_trainl = ytrainl[:,:,:2]

# Utilise -1 pour que NumPy calcule automatiquement la dimension finale (n_sortie * 2)
y_trainh64 = y_trainh.reshape(y_trainh.shape[0], -1)

"""
X_val = np.array([[vin3[i][j] for j in range(1, 4)] for i in range(len(vin3))])
y_val1 = np.array([[v1[i][j] for j in range(1, 3)] for i in range(len(v1))]) # point de la sonde 1
y_val2 = np.array([[v2[i][j] for j in range(1, 3)] for i in range(len(v2))]) # point de la sonde 2
y_val3 = np.array([[v3[i][j] for j in range(1, 3)] for i in range(len(v3))]) # point de la sonde 3

modeleh = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(
        max_depth = 200,
        random_state= SEED,
        n_jobs= -1
    )
)

modeleh.fit(X_train, ytrainh)
print(modeleh.predict(X_train[0]))"""