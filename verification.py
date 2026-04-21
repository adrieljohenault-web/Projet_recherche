# Ce programme a pour but d'évaluer l'efficacité et la précision de notre fonction de transfert réalisée sans apprentissage automatique. Ses résultats seront par la suite utilisés pour être comparés avec ceux de la fonction de transfert réalisée à l'aide de cette approche.

###TO DO###

# Il ne reste plus qu'à compléter la fonction MSE en prenant en compte le niveau d'eau.


import numpy as np
import os
from variables_globales import *
from formatage_verif import *
from Fonction_de_transfert import OS2NS, distance_euclidienne
from Schéma_bathymétrie import bathy

# Choix des 4 points les plus proches de chaque sonde pour réaliser une interpolation linéaire et choisir un point moyen qui les représentera.

d_sonde = [[0. for _ in range(n_sortie)] for _ in range(3)]

for k in range(n_sortie):
    for i in range(3):
        d_sonde[i][k] = distance_euclidienne([bathy[k][0], bathy[k][1]], sondes[i])

points_inter = [[[] for _ in range(4)] for _ in range(3)]

for i in range(3):
    for j in range(4):
        closest_index = np.argmin(d_sonde[i])
        closest_distance = np.min(d_sonde[i])
        #closest_point = np.array([bathy[closest_index][0], bathy[closest_index][1]])
        points_inter[i][j] = [closest_index, closest_distance]
        d_sonde[i][closest_index] = 10**10


def correspondance_maree(date: Date, table_marees = table_maree) -> float:
    for x in table_marees:
        if (x[2] == date._annee and
            x[1] == date._mois and
            x[0] == date._jour and
            x[3] == date._h):
            h = x[5]
            return (h-min_h)/(max_h-min_h)

def correspondance_es(sortie: list, entree: list = vin):
    """Prend pour chaque heure de la liste entree une donnée de la liste sortie la plus proche possible et supérieure à l'heure en question, et retourne une liste des indices à utiliser comme masque pour obtenir des listes de sortie qui correspondent à la liste des entrées.
    Données d'entrée : Hs, Tp, Dir
    Données de sortie : Hs, Tp"""
    
    indices = []

    for i in range(len(entree)):
        for j in range(len(sortie)):
            date1 = Date(entree[i][0][0],entree[i][0][1],entree[i][0][2],entree[i][0][3],entree[i][0][4])
            date2 = Date(sortie[j][0][0],sortie[j][0][1],sortie[j][0][2],sortie[j][0][3],sortie[j][0][4])

            if compare(date1, date2): 
                indices.append([i, j])
                break
    mask_entree = [False for _ in range(len(entree))]
    mask_sortie = [False for _ in range(len(sortie))]
    i_entree = [indices[k][0] for k in range(len(indices))]
    i_sortie = [indices[k][1] for k in range(len(indices))]

    for index in i_entree :
        mask_entree[index] = True

    for index in i_sortie :
        mask_sortie[index] = True

    entree = np.asarray(entree, dtype = 'object')
    sortie = np.asarray(sortie, dtype = 'object')

    return entree[mask_entree], sortie[mask_sortie]

r_data0 = correspondance_es(v2)
n_data = len(r_data0[0])

r_data = [np.zeros((n_data, 3)), np.zeros((n_data, 2))]
dates = []

for k in range(2):
    for i in range(n_data):
        for j in range(3-k):
            r_data[k][i][j] = r_data0[k][i][j+1]

for i in range(n_data):
    dates.append(r_data0[0][i][0])
#modele = OS2NS(, .5, True)
# Confrontation aux données obtenues grâce OS2NS par calcul de la MSE


def MSE(modele: list, r_data: list[list, list]):
    """Prend en argument deux listes contenant chacune des données d'entrée et de sortie, et calcule la MSE."""

    # 1. Importer les données de OS2NS au niveau des points concernés


