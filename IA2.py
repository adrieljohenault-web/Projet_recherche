# Le résonnement est le suivant. Si on suppose que Delft3D dit la vérité,
# c'est que le problème vient de comment on calcul le niveau de la marée. 


# le but est de construire ^y = a * Sh + b * Sl, avec a,b > 0 des vecteurs de dimenssion 2

# imports

import os 
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import csv
from scipy.optimize import nnls
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from variables_globales import *
from Fonction_de_transfert import *
from sonde_donnee_formatage import v1, v2, v3, vin,vin_sync



#### Étape 1 ####
# L'objectif ici est de construire les vecteurs d'apprentissage X. 
# définiton des entrées X
# X = [Hs_large, Tp_large, Dir_large, marnage, depth]
# Hs_large : la valeur de Hs au large
# Tp_large : la valeur de Tp au large
# Dir_large : la valeur de la direction au large
# marnage : la valeur entre marée haute et marée basse. On a observer que toutes les marées hautes n'avait pas la même hauteurs d'eau 
# depth : la profondeur moyenne du point de la sonde

# la taille de X est 7440 * 3, 7440 : nombre de points de données pour les 3 sondes, 3 : nombre de sondes
# X = [sonde1, sonde2, sonde3]

# calcul des depth

def get_mean_depth(v_sonde):
    depth = v_sonde[:,3]
    return np.mean(depth)

depth_v1 = get_mean_depth(v1)
depth_v2 = get_mean_depth(v2)
depth_v3 = get_mean_depth(v3)

# calcul du marnage 

def compute_marnage(v_large, target_dates, window_days=2):
    source_dates = v_large[:, 0]
    dpts = v_large[:, 4].astype(float)
    
    marnage_source = []
    for i, date in enumerate(source_dates):
        cutoff = date - timedelta(days=window_days)
        mask = (source_dates >= cutoff) & (source_dates <= date)
        
        if np.any(mask):
            val = dpts[mask].max() - dpts[mask].min()
            marnage_source.append(val)
        else:
            marnage_source.append(np.nan)
    
    marnage_source = np.array(marnage_source)

    indices = np.searchsorted(source_dates, target_dates)
    marnage_interp = []

    for i, date in enumerate(target_dates):
        k = indices[i]

        if k == 0 or k >= len(v_large):
            marnage_interp.append(np.nan)
            continue

        dt_inf = source_dates[k-1]
        dt_sup = source_dates[k]
        
        delta = (dt_sup - dt_inf).total_seconds()
        w_sup = (date - dt_inf).total_seconds() / delta
        w_inf = 1 - w_sup

        m_val = w_inf * marnage_source[k-1] + w_sup * marnage_source[k]
        marnage_interp.append(m_val)
    
    # Retourne un array avec [Date, Valeur_Marnage]
    return np.column_stack((target_dates, marnage_interp))

marnage = compute_marnage(vin, vin_sync[:, 0])



# création du vecteur X

def get_X(vin_sync, marn_data, depth1, depth2, depth3):
    mask = (vin_sync[:, 0] == marn_data[:, 0])
    # On ne garde que les lignes valides
    vin_sync_clean = vin_sync[mask]
    marn_val_clean = marn_data[mask, 1]
    
    # Optionnel : Alerte si des dates ont été supprimées
    diff = len(vin_sync) - len(vin_sync_clean)
    if diff > 0:
        print(f"Attention : {diff} lignes supprimées car les dates ne coïncidaient pas.")

    # 2. Construction du bloc commun : [Date, Hs, Tp, Dir, Marnage]
    # On utilise column_stack pour fusionner les colonnes proprement
    common = np.column_stack((vin_sync_clean, marn_val_clean))

    # 3. Ajout de la profondeur pour chaque sonde
    n = len(common)
    X1 = np.column_stack((common, np.full(n, depth1)))
    X2 = np.column_stack((common, np.full(n, depth2)))
    X3 = np.column_stack((common, np.full(n, depth3)))

    # 4. Empilement vertical final
    return np.vstack((X1, X2, X3))

# Utilisation
X_dates = get_X(vin_sync, marnage, depth_v1, depth_v2, depth_v3)


#### Étape 2 ####
# L'objectif est de construire les y = (a*,b*) désirée. cad, qu'on cherche a*,b* qui minimisent la norme de a*(SH,i) + b*(SL,i) - y,i pour toutes les dates i
# pour lesquelles on a des données des sondes

# coef = [a,b]      a: S_H, b:S_L

def get_hyp_ab(v_sonde, points, vin_sync):

    # Calcule σ_Hs et σ_Tp une seule fois sur l'ensemble du dataset
    sigma_Hs = np.std(v_sonde[:, 1].astype(float))
    sigma_Tp = np.std(v_sonde[:, 2].astype(float))
    D = np.array([1.0 / sigma_Hs, 1.0 / sigma_Tp])  # vecteur diagonal de D

    n_points = v_sonde.shape[0]
    arr_coefs = np.zeros((n_points, 3), dtype=object)

    for i in range(n_points):           #n_points
        if v_sonde[i, 0] != vin_sync[i, 0]:
            print(f"Désynchronisation détectée à l'index {i}")
            continue

        y_i = v_sonde[i, 1:3].astype(float)

        Hs, Tp, Dir = vin_sync[i, 1], vin_sync[i, 2], vin_sync[i, 3]
        pred_low, pred_high = OS2NS_uni_pluriel(points, Hs, Tp, Dir, 0, False)

        S_H_i = pred_high[:, :2].astype(float).flatten()
        S_L_i = pred_low[:, :2].astype(float).flatten()

        A = np.column_stack([S_H_i, S_L_i])

        A_scaled = A * D[:, np.newaxis]   # shape (2, 2)
        y_scaled = y_i * D                # shape (2,)


        coeffs, _ = nnls(A_scaled, y_scaled)

        print(coeffs[0], coeffs[1])

        arr_coefs[i, 0] = v_sonde[i, 0]
        arr_coefs[i, 1] = coeffs[0]    # Coefficient a (S_H)
        arr_coefs[i, 2] = coeffs[1]    # Coefficient b (S_L)
    
    return arr_coefs


def save_coefs(arr_coefs, filepath):
    """Sauvegarde arr_coefs [[datetime, a, b], ...] en CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "coef_a", "coef_b"])  # en-tête
        for row in arr_coefs:
            writer.writerow([
                row[0].strftime("%Y-%m-%d %H:%M"),  # datetime → string ISO
                float(row[1]),
                float(row[2])
            ])
    print(f"Coefficients sauvegardés → {filepath}")


def load_coefs(filepath):
    """Recharge un fichier CSV sauvegardé par save_coefs."""
    result = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append([
                datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M"),
                np.float64(row["coef_a"]),
                np.float64(row["coef_b"])
            ])
    return np.array(result, dtype=object)


def get_hyp(v1, v2, v3, vin_sync, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    Y1 = get_hyp_ab(v1, points_and_weights[0], vin_sync)
    Y2 = get_hyp_ab(v2, points_and_weights[1], vin_sync)
    Y3 = get_hyp_ab(v3, points_and_weights[2], vin_sync)

    save_coefs(Y1, os.path.join(output_dir, "coefs_sonde1.csv"))
    save_coefs(Y2, os.path.join(output_dir, "coefs_sonde2.csv"))
    save_coefs(Y3, os.path.join(output_dir, "coefs_sonde3.csv"))

    return Y1, Y2, Y3

def get_y():
    Y1 = load_coefs(os.path.join(".", "coefs_sonde1.csv"))
    Y2 = load_coefs(os.path.join(".", "coefs_sonde2.csv"))
    Y3 = load_coefs(os.path.join(".", "coefs_sonde3.csv"))

    return np.vstack((Y1,Y2,Y3))

y_dates = get_y()

#### Étape 3 ####
# L'objectif de cette étape est de mettre en place le modèle d'apprentissage f(X) = y


def get_X_y_wo_date(X,y):
    for k in range(X.shape[0]):
        if X[k][0] != y[k][0]:
            print(k)
    return X[:, 1:], y[:, 1:]

def get_IA2():
    X,y = get_X_y_wo_date(X_dates,y_dates)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Définition du modèle de base XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,      # Nombre d'arbres
        learning_rate=0.05,     # Vitesse d'apprentissage
        max_depth=7,           # Profondeur des arbres
        subsample=0.8,         # Fraction des données utilisées par arbre
        colsample_bytree=0.8,  # Fraction des colonnes utilisées par arbre
        objective='reg:squarederror',
        random_state=42, 
        n_jobs=2
    )

    multi_model = MultiOutputRegressor(xgb_model)

    print("Début de l'entraînement...")
    multi_model.fit(X_train, y_train)
    print("Entraînement terminé.")

    #prediction
    y_pred = multi_model.predict(X_test)

    #Évaluation des performances
    for i in range(y.shape[1]):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"\n--- Cible y{i+1} ---")
        print(f"Erreur Quadratique Moyenne (MSE) : {mse:.4f}")
        print(f"Score R2 (Précision) : {r2:.4f}")
#get_IA2()