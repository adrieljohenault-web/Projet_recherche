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
