# Ce programme a pour but d'évaluer l'efficacité et la précision de notre fonction de transfert réalisée sans apprentissage automatique. Ses résultats seront par la suite utilisés pour être comparés avec ceux de la fonction de transfert réalisée à l'aide de cette approche.

import numpy as np
import os
from variables_globales import *
from formatage_verif import *
from Fonction_de_transfert import OS2NS



def correspondance(sortie: list, entree: list = vin):
    """Prend pour chaque heure de la liste entree une donnée de la liste sortie la plus proche possible et supérieure à l'heure en question, et retourne une liste des indices à utiliser comme masque pour obtenir des listes de sortie qui correspondent à la liste des entrées."""
    
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


print(correspondance(v1))

# Confrontation aux données obtenues grâce OS2NS par calcul de la MSE


def MSE(modele: list[list, list], r_data: list[list, list]):
    """Prend en argument deux listes contenant chacune des données d'entrée et de sortie, et calcule la MSE."""

    # 1. Importer les données de OS2NS au niveau des points concernés
