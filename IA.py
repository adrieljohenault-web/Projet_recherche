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
import pickle


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


X_val = np.array([[vin3[i][j] for j in range(1, 4)] for i in range(len(vin3))])
y_val1 = np.array([[v1[i][j] for j in range(1, 3)] for i in range(len(v1))]) # point de la sonde 1
y_val2 = np.array([[v2[i][j] for j in range(1, 3)] for i in range(len(v2))]) # point de la sonde 2
y_val3 = np.array([[v3[i][j] for j in range(1, 3)] for i in range(len(v3))]) # point de la sonde 3

"""
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

#variable d'apprentissage
BLOCK_SIZE = 500
n_outputs = y_trainh64.shape[1]

# ---- DOSSIER DE SAUVEGARDE ----

save_dir = os.path.join(path, "RandomForest_AI")
os.makedirs(save_dir, exist_ok=True)


# ---- ENTRAÎNEMENT ----

def training():
    for start in range(0, n_outputs, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_outputs)
        save_path = os.path.join(save_dir, f"model_block_{start}_{end}.pkl")

        if os.path.exists(save_path):
            print(f"Block {start}:{end} already exists, skipping.")
            continue

        print(f"Training block {start}:{end}...")
        y_block = y_trainh64[:, start:end]

        model_block = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=50, max_depth=20,
                                max_features=2, n_jobs=2, random_state=SEED)
        )
        model_block.fit(X_train, y_block)

        with open(save_path, "wb") as f:
            pickle.dump(model_block, f)
        print(f"  -> Saved.")


# ---- PRÉDICTION ----

def predict_all(X, path, block_size=500, n_outputs=72000):
    save_dir = os.path.join(path, "RandomForest_AI")
    parts = []
    for start in range(0, n_outputs, block_size):
        end = min(start + block_size, n_outputs)
        save_path = os.path.join(save_dir, f"model_block_{start}_{end}.pkl")

        with open(save_path, "rb") as f:
            model_block = pickle.load(f)

        parts.append(model_block.predict(X))

    return np.concatenate(parts, axis=1)


y_pred = predict_all(X_train, path, block_size=BLOCK_SIZE, n_outputs=y_trainh64.shape[1])