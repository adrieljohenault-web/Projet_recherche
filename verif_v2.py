# l'objectif de ce fichier est de fournir une comparaison entre les fonctions toruver par la fonction de transfert et les données réelles
#import

import os 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv

from variables_globales import *
from Fonction_de_transfert import *


# ---------- Importation des données de sortie mesurées ---------
# verif_k = [YYYY, MM, DD, hh, mm, ss, h[m], Hm0[m], Hs[m], Tm[s], Tp[s]] 

with open(os.path.join(path, "Capteurs_pression", "S1_recup_capteur-haut_2012-12-13_2013-03-12_waveStats_filt_h01.1.dat"), 'r'
) as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_1 = np.array(lines)
verif_1 = verif_1[1:]

with open(os.path.join(path, "Capteurs_pression", "S2_recup_capteur-bas_2012-12-13_2013-03-12_waveStats_filt_h01.1.dat"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_2 = np.array(lines)
verif_2 = verif_2[7:-5]

with open(os.path.join(path, "Capteurs_pression", "S3_capteur_offshore_2012-12-11_2013-03-14_waveStats_filt_h01.1.dat"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_3 = np.array(lines)
verif_3 = verif_3[300:-311]


#on va mettre dans un format plus simple à comprendre 
# ( pour v3, on garde la valeur de dpt car elle va nous permettre de calculer le coefiscient des marées )

def format_sonde(verif: np.ndarray) -> np.ndarray:
    dts = np.array([
        datetime.datetime(int(row[0]), int(row[1]), int(row[2]),
                 int(row[3]), int(row[4]))
        for row in verif
    ], dtype=object)

    Hs  = verif[:, 8].astype(float)
    Tp  = verif[:, 10].astype(float)
    dpt = verif[:, 6].astype(float)

    return np.column_stack([dts, Hs, Tp, dpt])

v1 = format_sonde(verif_1)
v2 = format_sonde(verif_2)
v3 = format_sonde(verif_3)
# le nouveau format des sondes est le suivant : [date, Hs,Tp, dt]

# obtention des coefiscient de la marée en fonction de la date

def get_extremum(hauteurs: list):
    hauteurs.sort()
    ind_5percent = int(np.ceil(len(hauteurs) * 0.05))
    high = np.mean(hauteurs[-ind_5percent:])
    low = np.mean(hauteurs[:ind_5percent])
    return high, low

def get_coef_marree(v_sonde:np.array):
    dpt = v_sonde[:,-1]
    dpt_sorted = sorted(dpt)

    high,low = get_extremum(dpt_sorted)

    arr_coef = np.array([[date, 0] for date in v_sonde[:,0]])

    for i in range(len(v_sonde)):
        coef = min(1,max(( dpt[i] - low ) / ( high - low ),0))
        arr_coef[i][1] = coef
    
    return arr_coef

v_marree = get_coef_marree(v3)

# on intersect les données pour qu'elles soient compatibles sur les dates

def synchroniser(*arrays: np.ndarray) -> tuple:
    """
    Ne garde que les lignes dont la date (colonne 0) est présente
    dans tous les arrays passés en argument.
    """
    # Ensemble des dates communes à tous les arrays
    dates_communes = set(arrays[0][:, 0])
    for arr in arrays[1:]:
        dates_communes &= set(arr[:, 0])

    # Filtrage de chaque array
    arrays_sync = tuple(
        arr[np.array([dt in dates_communes for dt in arr[:, 0]])]
        for arr in arrays
    )
    return arrays_sync

v1, v2, v3, v_marree = synchroniser(v1, v2, v3, v_marree)

# ---------- Importation des données de mesurées au large ---------

with open(os.path.join(path, "Vagues_forcage", "Waves_resourcecode_138311.csv"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

vin = []
for line in lines[1:]:          # on saute le header
    date_str = line[0]             # '2009-12-31'
    champs   = line[1].split(',')  # split par virgule
    
    time_str = champs[0]           # '23:00:00'
    
    dt = datetime.datetime(
        int(date_str[:4]),          # année
        int(date_str[5:7]),         # mois
        int(date_str[8:10]),        # jour
        int(time_str[:2]),          # heure
        int(time_str[3:5])          # minute
    )
    
    Hs  = float(champs[2])
    Tp  = float(champs[5])
    Dir = float(champs[7])
    
    vin.append([dt, Hs, Tp, Dir])

vin = np.array(vin[25862:27996], dtype=object)

def coincide_large_to_shore(dates: np.ndarray, vin: np.ndarray) -> np.ndarray:

    vin_dates = vin[:, 0]  # array de datetime

    # np.searchsorted trouve pour chaque date l'indice k tel que
    # vin_dates[k-1] <= date < vin_dates[k]
    indices = np.searchsorted(vin_dates, dates)

    vin_new = []
    for i, date in enumerate(dates):
        k = indices[i]

        # Gardes-fous aux bords
        if k == 0 or k >= len(vin):
            continue  # pas de données encadrantes, on saute

        dt_inf = vin[k-1][0]
        dt_sup = vin[k][0]
        delta  = (dt_sup - dt_inf).total_seconds()

        w_sup = (date - dt_inf).total_seconds() / delta
        w_inf = 1 - w_sup

        Hs_new  = w_inf * float(vin[k-1][1]) + w_sup * float(vin[k][1])
        Tp_new  = w_inf * float(vin[k-1][2]) + w_sup * float(vin[k][2])
        Dir_new = w_inf * float(vin[k-1][3]) + w_sup * float(vin[k][3])

        vin_new.append([date, Hs_new, Tp_new, Dir_new])

    return np.array(vin_new, dtype=object)

# fait en sorte d'avoir les valeurs au large qui coincide en date avec les valeurs des sondes
vin_sync = coincide_large_to_shore(v1[:, 0], vin)

# Après coincide_large_to_shore, re-synchroniser
v1, v2, v3, v_marree, vin_sync = synchroniser(v1, v2, v3, v_marree, vin_sync)


def get_error(v_sonde, num_sonde, vin_sync, v_marree):
    closest = points_and_weights[num_sonde]
    list_points = [int(ind) for ind, w in closest]
    weights     = [w        for ind, w in closest]

    error = []

    for i in range(540,541):                  # len(v_sonde)
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

    for i in range(540,541):                  # len(v1)
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

sauvegarde_error(v1, v2, v3, vin_sync, v_marree)







