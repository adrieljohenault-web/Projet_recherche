# Ce programme vise à rendre utilisables les données des 4 sondes pour vérification.

import numpy as np
import os
from variables_globales import *
import Fonction_de_transfert
import csv
from datetime import datetime

class Date():
    def __init__(self, annee: int, mois: int, jour: int, h: int, m: int):
        self._annee = annee
        self._mois = mois
        self._jour = jour
        self._h = h
        self._m = m
    
    def __get__(self):
        return [self._annee, self._mois, self._jour, self._h, self._m]
    
def compare(date1: Date, date2: Date):
    return (
        date1._annee == date2._annee and
        date1._mois == date2._mois and
        date1._jour == date2._jour and
        date1._h == date2._h
    )

# Importation des données de sortie mesurées

with open(os.path.join(path, "Capteurs_pression", "S1_recup_capteur-haut_2012-12-13_2013-03-12_waveStats_filt_h01.1.dat"), 'r'
) as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_1 = lines
verif_1 = verif_1[1:]

with open(os.path.join(path, "Capteurs_pression", "S2_recup_capteur-bas_2012-12-13_2013-03-12_waveStats_filt_h01.1.dat"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_2 = lines
verif_2 = verif_2[7:-5] # Faire commencer et finir le bon jour à la bonne heure

with open(os.path.join(path, "Capteurs_pression", "S3_capteur_offshore_2012-12-11_2013-03-14_waveStats_filt_h01.1.dat"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_3 = lines
verif_3 = verif_3[300:-311] # Faire commencer et finir le bon jour à la bonne heuref

# Importation des données d'entrée mesurées

with open(os.path.join(path, "Vagues_forcage", "Waves_resourcecode_138311.csv"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_in = lines

verif_in = verif_in[25863:27996] # Troncature de verif_in qui va de 2009 à 2020

# Formatage de verif_in

vin0 = []

for donnees in verif_in:
    donnees[0] = donnees[0].split('-')

    donnees[1] = donnees[1].split(',')
    donnees[1][0] = donnees[1][0].split(':')
    donnees[1][0].pop(2)
    vin0.append([donnees[0][j] for j in range(3)]+[donnees[1][0][j] for j in range(2)]+[donnees[1][j] for j in range(1, 14)])

vin0 = np.array(vin0)

# Extraction de dpt
vin_dpt = []
for i in range(len(vin0)):
    date = Date(int(vin0[i][0]), int(vin0[i][1]), int(vin0[i][2]), 
                int(vin0[i][3]), int(vin0[i][4]))
    vin_dpt.append([date.__get__(), float(vin0[i][5])])

# Retrait des données inutiles dans les listes d'entrée

def formatage(verif: list):
    for donnees in verif:
        donnees.pop(5)
        donnees.pop(5)
        donnees.pop(5)
        donnees.pop(6)

# Retrait des données inutiles dans l'entrée (dpt, t02, tm01, lm, dp, spr, cge, uwnd, vwnd, ucur, vcur) >> On garde hs, tp, dir

numbers = [5, 7, 8, 10, 12, 13, 14, 15, 16, 17]
mask = [True for _ in range(18)]

for i in numbers:
    mask[i] = False

mask = np.array(mask)

vin1 = []

for donnees in vin0:
    donnees = donnees[mask]
    vin1.append(donnees)

formatage(verif_1)
formatage(verif_2)
formatage(verif_3)

# Les données ne se correspondent pas en dates et en heures, nous allons donc créer une fonction de correspondance qui prend en argument une liste de vérification d'entrée et une de sortie et qui prend pour chaque heure de la liste d'entrée une donnée de la liste de sortie, la plus proche possible de l'heure prise.

# Allure des données d'entrée : ['2013' '03' '12' '08' '00' '2.834' '8.849557522123893' '21.6']
# Annee, mois, jour, heure, minute, hs, tp, dir
# Allure des données de sortie : ['2012', '12', '13', '13', '14', '0.49745', '12.50000']
# Annee, mois, jour, heure, minute, Hs, Tp

# Transformation des string en int

def str_to_num(verif):
    for h in range(len(verif)):
        for i in range(len(verif[0])):
            if i < 5:
                verif[h][i] = int(verif[h][i])
            else: verif[h][i] = float(verif[h][i])

vin2 = [[vin1[i][j] for j in range(len(vin1[0]))] for i in range(len(vin1))]

str_to_num(verif_1)
str_to_num(verif_2)
str_to_num(verif_3)
str_to_num(vin2)

# Données mesurées au large ; formatage définitif

vin = [[] for _ in range(len(vin1))]

for i in range(len(vin2)):
    date = Date(vin2[i][0], vin2[i][1], vin2[i][2], vin2[i][3], vin2[i][4])
    vin[i] = [date.__get__()]
    for j in range(5, len(vin2[0])): vin[i].append(vin2[i][j])

# Données mesurées pour les trois sondes ; formatage définitif

v1 = [[] for _ in range(len(verif_1))]

for i in range(len(verif_1)):
    date = Date(verif_1[i][0], verif_1[i][1], verif_1[i][2], verif_1[i][3], verif_1[i][4])
    v1[i] = [date.__get__()]
    for j in range(5, len(verif_1[0])): v1[i].append(verif_1[i][j])

v2 = [[] for _ in range(len(verif_2))]

for i in range(len(verif_2)):
    date = Date(verif_2[i][0], verif_2[i][1], verif_2[i][2], verif_2[i][3], verif_2[i][4])
    v2[i] = [date.__get__()]
    for j in range(5, len(verif_2[0])): v2[i].append(verif_2[i][j])

v3 = [[] for _ in range(len(verif_3))]

for i in range(len(verif_3)):
    date = Date(verif_3[i][0], verif_3[i][1], verif_3[i][2], verif_3[i][3], verif_3[i][4])
    v3[i] = [date.__get__()]
    for j in range(5, len(verif_3[0])): v3[i].append(verif_3[i][j])

vin3 = [[] for _ in range(len(vin2))]

for i in range(len(vin2)):
    date = Date(vin2[i][0], vin2[i][1], vin2[i][2], vin2[i][3], vin2[i][4])
    vin3[i] = [date.__get__()]
    for j in range(5, len(vin2[0])): vin3[i].append(vin2[i][j])

# Création d'une table des marées

with open(
    os.path.join(path, "Vagues_forcage", "Tide_Brignogan_2009-2020_UTC_hourly.txt"),
    "r",
) as file:
    lines = file.readlines()[1:]
for i in range(len(lines)):
    lines[i] = lines[i].split()

table_maree = lines[25859:27993]

for i in range(len(table_maree)):
    for j in range(5):
        table_maree[i][j] = int(table_maree[i][j])
    table_maree[i][5] = float(table_maree[i][5])

hauteurs = [table_maree[i][5] for i in range(len(table_maree))]

def get_extremum(hauteurs: list):
    #on définit le high (le low) comme les 5 pourcent des valeurs les plus hautes (les plus basses)
    hauteurs.sort()
    ind_5percent = int(np.ceil(len(hauteurs) * 0.05))
    high = np.mean(hauteurs[-ind_5percent:])
    low = np.mean(hauteurs[:ind_5percent])
    return high, low

dpt_max,dpt_min = get_extremum([line[1] for line in vin_dpt])

def get_coef_marree(h:float,max_h,min_h):
    coef = ( h - min_h ) / ( max_h - min_h )
    if coef > 1 :
        return 1
    elif coef < 0 : 
        return 0
    else :
        return coef

dates = [line[0] for line in vin3]

#print(vin3[0], vin3[-1])
#print(v1[0], v1[-1])
#print(v2[0], v2[-1])
#print(v3[0], v3[-1])
#[2012, 12, 13, 13, 0], 2.06, 16.39344262295082, 283.2]

def get_closest_value(date1: list, list_condition_au_large : list):
    #date1 correspond à [annee, mois, jour, heure, minute], tandis que dates correspond à dates
    #le but de cette fonction est de trouver les poids et les indices qui s'approchent le plus de date1 dans dates (on se limite au 2 plus proche)
    #l'existence de données à moins d'1 heure dans la liste dates est donné par la 2ème fonction dates_verif() du README

    dates = [line[0] for line in list_condition_au_large]

    trouver = False
    for k in range(len(dates)-1):
        if dates[k][0:4]==date1[0:4]:
                trouver = True 

                min = date1[-1]
                weight_moins = (60-min)/60
                weight_plus = min/60

                
                val = [list_condition_au_large[k][1]*weight_moins + list_condition_au_large[k+1][1]*weight_plus,
                       list_condition_au_large[k][2]*weight_moins + list_condition_au_large[k+1][2]*weight_plus,
                       list_condition_au_large[k][3]*weight_moins + list_condition_au_large[k+1][3]*weight_plus]
                
                return val 
    return trouver

from datetime import datetime

def date_to_datetime(date: list) -> datetime:
    """Convertit [annee, mois, jour, heure, minute] en objet datetime."""
    return datetime(date[0], date[1], date[2], date[3], date[4])

def get_closest_value_2(date1: list, list_condition: list, seuil_minutes: float = 60.0):
    """
    Trouve la valeur dans list_condition dont la date est la plus proche de date1.
    
    - date1 : [annee, mois, jour, heure, minute]
    - list_condition : liste de [date, val1, val2, ...] avec date = [annee, mois, jour, heure, minute]
    - seuil_minutes : si le point le plus proche est à plus de seuil_minutes, retourne None
    
    Retourne les valeurs interpolées entre les deux entrées encadrantes si possible,
    sinon la valeur du point le plus proche.
    """
    dt1 = date_to_datetime(date1)

    # Calcul de l'écart en minutes pour chaque entrée
    ecarts = [(abs((date_to_datetime(ligne[0]) - dt1).total_seconds()) / 60, i)
              for i, ligne in enumerate(list_condition)]
    ecarts.sort()

    ecart_min, idx_proche = ecarts[0]

    if ecart_min > seuil_minutes:
        return None  # Pas de donnée suffisamment proche

    # Chercher les deux entrées qui encadrent date1 pour interpoler
    avant = None   # entrée juste avant dt1
    apres = None   # entrée juste après dt1

    for i, ligne in enumerate(list_condition):
        dt = date_to_datetime(ligne[0])
        if dt <= dt1:
            if avant is None or dt > date_to_datetime(list_condition[avant][0]):
                avant = i
        if dt >= dt1:
            if apres is None or dt < date_to_datetime(list_condition[apres][0]):
                apres = i

    # Si on a deux points encadrants, on interpole
    if avant is not None and apres is not None and avant != apres:
        dt_avant = date_to_datetime(list_condition[avant][0])
        dt_apres = date_to_datetime(list_condition[apres][0])

        duree_totale = (dt_apres - dt_avant).total_seconds()
        duree_avant  = (dt1 - dt_avant).total_seconds()

        w_apres = duree_avant / duree_totale   # poids du point après
        w_avant = 1 - w_apres                  # poids du point avant

        n_vals = len(list_condition[avant]) - 1  # nombre de valeurs (hors date)
        return [
            list_condition[avant][j+1] * w_avant + list_condition[apres][j+1] * w_apres
            for j in range(n_vals)
        ]

    # Sinon, retour de la valeur la plus proche sans interpolation
    return list_condition[idx_proche][1:]


def verif_modele_sonde(v_sonde:list, v_au_large:list, num_sonde:int, list_dpt:list):

    Lerreur = [[v_sonde[0],0,0] for _ in range(len(v_sonde))]
    list_points = [points_and_weights[num_sonde][j][0] for j in range(4)]

    with open(f"sonde_{num_sonde}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["date", "Hs_err", "Tp_err"])

        for k in range(6):               # len(v_sonde)
            date1 = v_sonde[k][0]
            val_large = get_closest_value(date1, v_au_large)
            Hs = val_large[0]
            Tp = val_large[1]
            Dir = val_large[2]

            dpt = get_closest_value_2(date1, list_dpt)

            coef = get_coef_marree(dpt[0], dpt_max, dpt_min)

            results = [
                Fonction_de_transfert.OS2NS_uni(list_points[j], Hs, Tp, Dir, coef, True)
                for j in range(4)
            ]

            Hs_val_fonction_transfert = sum(
                results[j][0] * points_and_weights[num_sonde][j][1] for j in range(4)
            )
            Tp_val_fonction_transfert = sum(
                results[j][1] * points_and_weights[num_sonde][j][1] for j in range(4)
            )
            Hs_sonde = v_sonde[k][1]
            Tp_sonde = v_sonde[k][2]

            Lerreur[k][1] = Hs_sonde - Hs_val_fonction_transfert
            Lerreur[k][2] = Tp_sonde - Tp_val_fonction_transfert
            writer.writerow([v_sonde[k][0], Lerreur[k][1], Lerreur[k][2]])
            print(k/len(v_sonde))

    return Lerreur

#verif_modele_sonde(v1,vin3,0,vin_dpt)
