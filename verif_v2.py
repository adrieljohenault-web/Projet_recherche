# l'objectif de ce fichier est de fournir une comparaison entre les fonctions toruver par la fonction de transfert et les données réelles
#import

import os 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv

from variables_globales import *
from Fonction_de_transfert import *
from err_suivant_coef_maree import plot_err_vs_coef_maree
from sonde_donnee_formatage import v1, v2, v3, v_marree, vin_sync

#calcul de l'erreur

def get_error(v_sonde, num_sonde, vin_sync, v_marree):
    closest = points_and_weights[num_sonde]
    list_points = [int(ind) for ind, w in closest]
    weights     = [w        for ind, w in closest]

    error = []

    for i in range(len(v1)):                  # len(v_sonde)
        Hs_large  = float(vin_sync[i][1])
        Tp_large  = float(vin_sync[i][2])
        Dir_large = float(vin_sync[i][3])
        coef      = float(v_marree[i][1])

        # Un seul appel pour les 4 points → shape (4, 8)
        sortie = OS2NS_vectorized_per_points2(
            Hs_large, Tp_large, Dir_large, coef, list_points
        )

        Hs_res = sum(sortie[j][0] * weights[j] for j in range(4))
        Tp_res = sum(sortie[j][1] * weights[j] for j in range(4))

        Hs_sonde = float(v_sonde[i][1])
        Tp_sonde = float(v_sonde[i][2])

        error.append((abs(Hs_res - Hs_sonde), abs(Tp_res - Tp_sonde)))

        print(i/len(v_sonde))

    return error

def get_error_all(v1,v2,v3, vin_sync, v_marree):
    closest1 = points_and_weights[0]
    closest2 = points_and_weights[1]
    closest3 = points_and_weights[2]

    list_points1 = [int(ind) for ind, w in closest1]
    list_points2 = [int(ind) for ind, w in closest2]
    list_points3 = [int(ind) for ind, w in closest3]
    weights1     = [w        for ind, w in closest1]
    weights2     = [w        for ind, w in closest2]
    weights3     = [w        for ind, w in closest3]

    error = []

    for i in range(len(v1)):                  # len(v1)
        dict = {}
        coef = float(v_marree[i][1])

        Hs_large  = float(vin_sync[i][1])
        Tp_large  = float(vin_sync[i][2])
        Dir_large = float(vin_sync[i][3])

        # sonde 1
        sortie1 = OS2NS_vectorized_per_points2(
            Hs_large, Tp_large, Dir_large, coef, list_points1
        )

        Hs_res1 = sum(sortie1[j][0] * weights1[j] for j in range(4))
        Tp_res1 = sum(sortie1[j][1] * weights1[j] for j in range(4))

        Hs_sonde1 = float(v1[i][1])
        Tp_sonde1 = float(v1[i][2])

        dict[1] = (abs(Hs_res1 - Hs_sonde1), abs(Tp_res1 - Tp_sonde1))

        # sonde 2
        sortie2 = OS2NS_vectorized_per_points2(
            Hs_large, Tp_large, Dir_large, coef, list_points2
        )

        Hs_res2 = sum(sortie2[j][0] * weights2[j] for j in range(4))
        Tp_res2 = sum(sortie2[j][1] * weights2[j] for j in range(4))

        Hs_sonde2 = float(v2[i][1])
        Tp_sonde2 = float(v2[i][2])

        dict[2] = (abs(Hs_res2 - Hs_sonde2), abs(Tp_res2 - Tp_sonde2))

        #sortie 3
        sortie3 = OS2NS_vectorized_per_points2(
            Hs_large, Tp_large, Dir_large, coef, list_points3
        )

        Hs_res3 = sum(sortie3[j][0] * weights3[j] for j in range(4))
        Tp_res3 = sum(sortie3[j][1] * weights3[j] for j in range(4))

        Hs_sonde3 = float(v3[i][1])
        Tp_sonde3 = float(v3[i][2])

        dict[3] = (abs(Hs_res3 - Hs_sonde3), abs(Tp_res3 - Tp_sonde3))

        print(i/len(v1))

        error.append(dict)

    return error


def sauvegarde_error(v1, v2, v3, vin_sync, v_marree, nom_fichier="erreurs_all_transfer_function.csv"):
    
    errors = get_error_all(v1, v2, v3, vin_sync, v_marree)

    with open(nom_fichier, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "date",
            "Hs_err_s1", "Tp_err_s1",
            "Hs_err_s2", "Tp_err_s2",
            "Hs_err_s3", "Tp_err_s3",
        ])

        for i, d in enumerate(errors):
            writer.writerow([
                v1[i][0],
                round(d[1][0], 4), round(d[1][1], 4),
                round(d[2][0], 4), round(d[2][1], 4),
                round(d[3][0], 4), round(d[3][1], 4),
            ])

    print(f"Sauvegardé : {nom_fichier} ({len(errors)} lignes)")


def get_MSE(nom_fichier="erreurs_all_transfer_function.csv"):
    """
    Calcule la Mean Squared Error (MSE) pour les 6 variables d'erreur stockées dans le CSV.
    Retourne un dictionnaire avec les valeurs de MSE pour chaque variable.
    """
    # On utilise numpy (déjà importé dans votre fichier) pour charger les données numériques.
    # On ignore la première ligne (header) et la première colonne (date).
    # Les colonnes d'erreur sont aux indices 1 à 6.
    try:
        data = np.genfromtxt(nom_fichier, delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5, 6))
        
        # Puisque le CSV contient déjà les écarts (erreurs), 
        # on calcule le carré de chaque valeur puis la moyenne de chaque colonne (axis=0).
        mse_values = np.mean(data**2, axis=0)
        
        # Noms des variables pour une lecture plus claire
        variables = ["Hs_s1", "Tp_s1", "Hs_s2", "Tp_s2", "Hs_s3", "Tp_s3"]
        
        # Création d'un dictionnaire pour associer chaque variable à sa MSE
        resultats = dict(zip(variables, mse_values))
        
        print("\n--- Résultats de la MSE ---")
        for var, val in resultats.items():
            print(f"MSE {var}: {val:.4f}")
            
        return resultats

    except Exception as e:
        print(f"Erreur lors du calcul de la MSE : {e}")
        return None


def hist_err_marr():
    plot_err_vs_coef_maree(
        v_marree,
        nom_fichier="erreurs_all_transfer_function.csv",
        n_bins=10,
        save_path="err_vs_coef_maree.png",   # optionnel
    )

def generer_illustrations_comparaison(v1, v2, v3, vin_sync, v_marree, save_prefix="transfer_vs_reel"):
    """
    Génère deux figures de comparaison pour les 3 sondes (Hs et Tp) :
    1. Une comparaison temporelle (Séries temporelles superposées)
    2. Un diagramme de dispersion (Scatter plot avec ligne d'identité 1:1)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    sondes_data = [v1, v2, v3]
    closest_points = [points_and_weights[0], points_and_weights[1], points_and_weights[2]]

    # Initialisation des structures pour stocker les prédictions de la fonction de transfert
    predictions = {
        1: {"Hs": [], "Tp": []},
        2: {"Hs": [], "Tp": []},
        3: {"Hs": [], "Tp": []}
    }
    dates = [row[0] for row in v1]
    n_points = len(v1)

    print("Calcul des prédictions de la fonction de transfert en cours...")
    for i in range(n_points):
        coef      = float(v_marree[i][1])
        Hs_large  = float(vin_sync[i][1])
        Tp_large  = float(vin_sync[i][2])
        Dir_large = float(vin_sync[i][3])

        for s_idx in [1, 2, 3]:
            closest = closest_points[s_idx - 1]
            list_points = [int(ind) for ind, w in closest]
            weights     = [w        for ind, w in closest]

            # Appel à votre fonction géométrique vectorisée
            sortie = OS2NS_vectorized_per_points2(
                Hs_large, Tp_large, Dir_large, coef, list_points
            )

            # Reconstitution par pondération des 4 points les plus proches
            Hs_res = sum(sortie[j][0] * weights[j] for j in range(4))
            Tp_res = sum(sortie[j][1] * weights[j] for j in range(4))

            predictions[s_idx]["Hs"].append(Hs_res)
            predictions[s_idx]["Tp"].append(Tp_res)

    print("Génération des graphiques...")

    # ==========================================
    # FIGURE 1 : SÉRIES TEMPORELLES COMPARATIVES
    # ==========================================
    fig_time, axes_time = plt.subplots(3, 2, figsize=(16, 12), sharex='col')
    
    for s_idx in [1, 2, 3]:
        v_sonde = sondes_data[s_idx - 1]
        Hs_reel = v_sonde[:, 1].astype(float)
        Tp_reel = v_sonde[:, 2].astype(float)
        
        Hs_pred = np.array(predictions[s_idx]["Hs"])
        Tp_pred = np.array(predictions[s_idx]["Tp"])
        
        row = s_idx - 1
        
        # Colonne gauche : Hs
        axes_time[row, 0].plot(dates, Hs_reel, label="Mesuré (Sonde)", color="black", alpha=0.7, lw=1.5)
        axes_time[row, 0].plot(dates, Hs_pred, label="Fonction de Transfert", color="crimson", linestyle="--", alpha=0.85, lw=1.5)
        axes_time[row, 0].set_ylabel(f"Hs (m) - Sonde {s_idx}", fontsize=11)
        axes_time[row, 0].grid(True, linestyle=":", alpha=0.6)
        axes_time[row, 0].legend(loc="upper right")
        if row == 0:
            axes_time[row, 0].set_title("Hauteur significative des vagues (Hs)", fontsize=14, fontweight='bold')
            
        # Colonne droite : Tp
        axes_time[row, 1].plot(dates, Tp_reel, label="Mesuré (Sonde)", color="black", alpha=0.7, lw=1.5)
        axes_time[row, 1].plot(dates, Tp_pred, label="Fonction de Transfert", color="royalblue", linestyle="--", alpha=0.85, lw=1.5)
        axes_time[row, 1].set_ylabel(f"Tp (s) - Sonde {s_idx}", fontsize=11)
        axes_time[row, 1].grid(True, linestyle=":", alpha=0.6)
        axes_time[row, 1].legend(loc="upper right")
        if row == 0:
            axes_time[row, 1].set_title("Période de pic (Tp)", fontsize=14, fontweight='bold')

    fig_time.autofmt_xdate()
    plt.tight_layout()
    fig_time.savefig(f"{save_prefix}_temporelle.png", dpi=300)
    plt.close(fig_time)

    # ==========================================
    # FIGURE 2 : DIAGRAMMES DE DISPERSION (1:1)
    # ==========================================
    fig_scat, axes_scat = plt.subplots(3, 2, figsize=(12, 14))
    
    for s_idx in [1, 2, 3]:
        v_sonde = sondes_data[s_idx - 1]
        Hs_reel = v_sonde[:, 1].astype(float)
        Tp_reel = v_sonde[:, 2].astype(float)
        
        Hs_pred = np.array(predictions[s_idx]["Hs"])
        Tp_pred = np.array(predictions[s_idx]["Tp"])
        
        row = s_idx - 1
        
        # Scatter Hs
        axes_scat[row, 0].scatter(Hs_reel, Hs_pred, alpha=0.4, color="crimson", edgecolors='none', s=20)
        lims_hs = [0, max(max(Hs_reel), max(Hs_pred)) * 1.05]
        axes_scat[row, 0].plot(lims_hs, lims_hs, 'k--', alpha=0.75, label="Ligne 1:1")
        axes_scat[row, 0].set_xlim(lims_hs)
        axes_scat[row, 0].set_ylim(lims_hs)
        axes_scat[row, 0].set_xlabel("Hs Mesuré (m)", fontsize=10)
        axes_scat[row, 0].set_ylabel(f"Hs Prédit (m) - Sonde {s_idx}", fontsize=10)
        axes_scat[row, 0].grid(True, linestyle=":", alpha=0.6)
        axes_scat[row, 0].legend(loc="upper left")
        if row == 0:
            axes_scat[row, 0].set_title("Dispersion Hs", fontsize=13, fontweight='bold')
            
        # Scatter Tp
        axes_scat[row, 1].scatter(Tp_reel, Tp_pred, alpha=0.4, color="royalblue", edgecolors='none', s=20)
        lims_tp = [0, max(max(Tp_reel), max(Tp_pred)) * 1.05]
        axes_scat[row, 1].plot(lims_tp, lims_tp, 'k--', alpha=0.75, label="Ligne 1:1")
        axes_scat[row, 1].set_xlim(lims_tp)
        axes_scat[row, 1].set_ylim(lims_tp)
        axes_scat[row, 1].set_xlabel("Tp Mesuré (s)", fontsize=10)
        axes_scat[row, 1].set_ylabel(f"Tp Prédit (s) - Sonde {s_idx}", fontsize=10)
        axes_scat[row, 1].grid(True, linestyle=":", alpha=0.6)
        axes_scat[row, 1].legend(loc="upper left")
        if row == 0:
            axes_scat[row, 1].set_title("Dispersion Tp", fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig_scat.savefig(f"{save_prefix}_dispersion.png", dpi=300)
    plt.close(fig_scat)

    print(f"Illustrations sauvegardées avec succès :\n - {save_prefix}_temporelle.png\n - {save_prefix}_dispersion.png")

generer_illustrations_comparaison(v1, v2, v3, vin_sync, v_marree, save_prefix="transfer_vs_reel")