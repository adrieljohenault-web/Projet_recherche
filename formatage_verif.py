# Ce programme vise à rendre utilisables les données des 4 sondes pour vérification.

import numpy as np
import os
from variables_globales import *

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

with open(os.path.join(path_adriel, "Capteurs_pression", "S1_recup_capteur-haut_2012-12-13_2013-03-12_waveStats_filt_h01.1.dat"), 'r'
) as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_1 = lines
verif_1 = verif_1[1:]

with open(os.path.join(path_adriel, "Capteurs_pression", "S2_recup_capteur-bas_2012-12-13_2013-03-12_waveStats_filt_h01.1.dat"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_2 = lines
verif_2 = verif_2[7:-5] # Faire commencer et finir le bon jour à la bonne heure

with open(os.path.join(path_adriel, "Capteurs_pression", "S3_capteur_offshore_2012-12-11_2013-03-14_waveStats_filt_h01.1.dat"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_3 = lines
verif_3 = verif_3[300:-311] # Faire commencer et finir le bon jour à la bonne heure

# Importation des données d'entrée mesurées

with open(os.path.join(path_adriel, "Vagues_forcage", "Waves_resourcecode_138311.csv"), 'r') as file :
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

verif_in = lines

verif_in = verif_in[25850:27995] # Troncature de verif_in qui va de 2009 à 2020

# Formatage de verif_in

vin0 = []

for donnees in verif_in:
    donnees[0] = donnees[0].split('-')

    donnees[1] = donnees[1].split(',')
    donnees[1][0] = donnees[1][0].split(':')
    donnees[1][0].pop(2)
    vin0.append([donnees[0][j] for j in range(3)]+[donnees[1][0][j] for j in range(2)]+[donnees[1][j] for j in range(1, 14)])

vin0 = np.array(vin0)

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

vin = [[] for _ in range(len(vin1))]

for i in range(len(vin2)):
    date = Date(vin2[i][0], vin2[i][1], vin2[i][2], vin2[i][3], vin2[i][4])
    vin[i] = [date.__get__()]
    for j in range(5, len(vin2[0])): vin[i].append(vin2[i][j])

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

with open(
    os.path.join(path_adriel, "Vagues_forcage", "Tide_Brignogan_2009-2020_UTC_hourly.txt"),
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
max_h = max(hauteurs)
min_h = min(hauteurs)