# l'objectif de ce fichier est de fournir une comparaison entre les fonctions toruver par la fonction de transfert et les données réelles
#import

import os 
import numpy as np
import datetime
import matplotlib.pyplot as plt

from variables_globales import *


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

def coincide_large_to_shore(dates:list, vin):
    vin_new = []
    
    for i in range(10,13):                      # len(dates)
        date = dates[i]

        date_sup = 0
        date_inf = 0
        finished = False
        k = 0
        while not finished:
            if vin[k][0] > date:
                date_sup = vin[k][0]
                date_inf = vin[k-1][0]
                finished = True
            else : 
                k+=1
        
        delta = date_sup - date_inf
        w_sup= (date - date_inf)/delta
        w_inf = (date_sup - date)/delta

        Hs_new = w_inf * vin[k-1][1] + w_sup * vin[k][1]
        Tp_new = w_inf * vin[k-1][2] + w_sup * vin[k][2]
        Dir_new = w_inf * vin[k-1][3] + w_sup * vin[k][3]

        print(vin[k-1], vin[k], v1[i], Hs_new, Tp_new, Dir_new)

        vin_new.append([date, Hs_new, Tp_new, Dir_new])

    
    return np.array(vin_new)

coincide_large_to_shore(v1[:,0], vin)


            






# sortie : ['2009-12-31', '23:00:00,31.5,2.002,5.0,6.74,7.462686567164178,75.0,6.1,6.0,49.3,13.9,-8.2,-6.9,-0.54,-0.36']
# format : [',dpt,hs,t02,t0m1,tp,lm,dir,dp,spr,cge,uwnd,vwnd,ucur,vcur']
