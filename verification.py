# Ce programme a pour but d'évaluer l'efficacité et la précision de notre fonction de transfert réalisée sans apprentissage automatique. Ses résultats seront par la suite utilisés pour être comparés avec ceux de la fonction de transfert réalisée à l'aide de cette approche.

###TO DO###

# Il ne reste plus qu'à compléter la fonction MSE en prenant en compte le niveau d'eau.
# Résoudre dans les problèmes d'indexations pour modele


import numpy as np
import os
from variables_globales import *
from formatage_verif import *
from Fonction_de_transfert import OS2NS, distance_euclidienne
from Schéma_bathymétrie import bathy

# Choix des 4 points les plus proches de chaque sonde pour réaliser une interpolation linéaire et choisir un point moyen qui les représentera.


def correspondance_maree(date: Date, table_marees = table_maree) -> float:
    for x in table_marees:
        if (x[2] == date._annee and
            x[1] == date._mois and
            x[0] == date._jour and
            x[3] == date._h):
            h = x[5]
            return ((h-min_h)/(max_h-min_h))

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

r_data = [np.zeros((n_data, 3)), np.zeros((n_data, 2))] # r_data0 sans les dates qui vont partir dans la liste dates

# Création de la liste des dates telle que dates[i] corresponde à la donnée i de n'importe quelle séries de données réelles

dates = []

for k in range(2):
    for i in range(n_data):
        for j in range(3-k):
            r_data[k][i][j] = r_data0[k][i][j+1]

offs = r_data[0]
nears = r_data[1]

for i in range(n_data):
    dates.append(r_data0[0][i][0])

# On se préoccupe de la sonde n°2

# Création de la liste d'entrée & passer en argument de la fonction MSE

modeles = np.zeros((3, n_data, 2)) # Hs et Tp pour chaque date, pour chaque sonde.

for j in range(3):
    for i in range(260): # ~6h, n_data prendrait 42 h...
        coef_maree_i = correspondance_maree(Date(dates[i][0], dates[i][1], dates[i][2], dates[i][3], dates[i][4]))
        Hs_i = offs[i][0]
        Tp_i = offs[i][1]
        Dir_i = offs[i][2]
        for k in range(4):
            pt_k = points_inter[j][k][0]
            dist_k = points_inter[j][k][1]
            Hs_ptk = OS2NS(Hs_i, Tp_i, Dir_i, coef_maree_i)[pt_k][0]
            Tp_ptk = OS2NS(Hs_i, Tp_i, Dir_i, coef_maree_i)[pt_k][1]
            modeles[j][i] += np.array([Hs_ptk, Tp_ptk])*dist_k
        modeles[j][i] = modeles[j][i]/somme_dist[j]
    
print(modeles)


# Confrontation aux données obtenues grâce OS2NS par calcul de la MSE


def MSE(modele: list, r_data: list[list, list]):
    """Prend en argument deux listes contenant chacune des données d'entrée et de sortie, et calcule la MSE."""




