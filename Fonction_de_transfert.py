import numpy as np
import os

### TO-D0 ###
# Comment faire pour prendre en compte les trois profils ?
# Remplacer Pos par la hauteur d'eau par souci de clarté
# Normaliser pour les distances euclidiennes
# blelbleblblble pas exactement mais ca retranscrit un pue l'effet de la phrase - Adriel Henault

# VARIABLES GLOBALES

n_valeurs_calc = 1761  # nombre de valeur d'entrée et de sortie qui ont été calculées par Delft3D (=len(entree))
n_interpolation = 5  # nombre de cas regardés pour l'interpolation
n_sortie = 36225  # nombre de lignes d'un fichier de sortie

# ouvrir les 3 blocks et les assembler dans entree de Delft3D

with open(
    "Fichier_d_encadrement/Lookup_table_Vougot/Delft3D_entrees/allblocks_Vougot.txt",
    "r",
) as file:
    lines = file.readlines()[1:]
for i in range(len(lines)):
    lines[i] = lines[i].split()

entree = lines

for i in range(len(entree)):
    entree[i] = list(map(float, entree[i]))  # convertir les string en float

entree = np.array(entree)


def nombre_fichier_sortie(n):
    nombre_de_charactere = len(f"{n}")
    ch = "0" * (4 - nombre_de_charactere)
    return ch + f"{n}"


# fonction de correspondance entre les sorties


def sortie_fichier(i: int) -> list:
    LSH = []
    ch = os.path.join(
        "Fichier_d_encadrement",
        "Lookup_table_Vougot",
        "Delft3D_sorties_gamma04",
        "SH",
        f"D3D_res{nombre_fichier_sortie(i)}_SH.txt",
    )
    with open(ch, "r") as file:
        LSH = file.readlines()[1:]
    for j in range(len(LSH)):
        LSH[j] = LSH[j].split()
        LSH[j] = [float(x) for x in LSH[j]]

    LSL = []
    ch = os.path.join(
        "Fichier_d_encadrement",
        "Lookup_table_Vougot",
        "Delft3D_sorties_gamma04",
        "SL",
        f"D3D_res{nombre_fichier_sortie(i)}_SL.txt",
    )
    with open(ch, "r") as file:
        LSL = file.readlines()[1:]
    for j in range(len(LSL)):
        LSL[j] = LSL[j].split()
        LSL[j] = [float(x) for x in LSL[j]]

    return np.array(LSL), np.array(LSH)


def distance_euclidienne(q1: list, q2: list):
    """Calcule la distance euclidienne entre les deux array de dimenison 4 q1 et q2"""
    distcarre = 0
    for k in range(len(q1)):
        distcarre += (q1[k] - q2[k]) ** 2
    return np.sqrt(distcarre)


def OS2NS(Hs: float, Tp: float, Dir: float, Pos: float, coef_maree: float, maree: bool) -> list:
    """Fonction de transfert en tant que telle : réalise l'interpolation qui permet d'obtenir les condtitions de déferlement. Retourne la matrice des conditions de déferlement en chaque point.
    Le paramètre coef_maree est un coefficient entre 0 et 1 permettant de tenir compte de la dépendance linéaire des conditions de sortie en fonction de la maree.
    Si maree est réglé sur True, alors la marée sera prise en compte et une seule valeur correspondante sera retournée. Dans le cas contraire, la fonction retourne deux valeurs de sortie correspondant respectivement aux marées basse et haute."""

    # Parcourir l'ensemble des données d'entrées et prendre les 5 plus petites distances euclidiennes à la situation en argument

    argument = np.array([Hs, Tp, Dir, Pos])
    five_closest = [[], [], [], [], []]
    memo: list[float] = np.array([0] * n_valeurs_calc)

    for i in range(n_valeurs_calc):
        memo[i] = distance_euclidienne(argument, entree[i])

    # Faire 5 minima en enlevant la valeur minimale à chaque fois ; stocker la distance et l'indice dans le tableau five_closest

    for i in range(n_interpolation):
        closest_index = int(np.argmin(memo))
        closest_distance = float(np.min(memo))
        closest_point = [float(entree[np.argmin(memo)][i]) for i in range(4)]
        five_closest[i] = [closest_distance, closest_index, closest_point]
        memo[np.argmin(memo)] = 1000000

    # Arriver aux données de sortie de Delft3D

    # sortie_low = [[tableau des conditions suivantes en chaque point du maillage] pour chaque condition d'entrée des five_closest] --> Hs, Tp, Tm01, Dp, Dm, DSpr, WD, Qb
    # sortie_high = ...

    sortie_low, sortie_high = [
        sortie_fichier(five_closest[i][1])[0] for i in range(n_interpolation)
    ], [sortie_fichier(five_closest[i][1])[1] for i in range(n_interpolation)]

    # Faire l'interpolation et retourner les valeurs finales

    sortieL, sortieH = np.array([[0] * 8 for _ in range(n_sortie)]), np.array([[0] * 8 for _ in range(n_sortie)])

    for i in range(n_sortie):
        for j in range(8):
            for h in range(n_interpolation):
                sortieL[i][j] += sortie_low[h][i][j] * five_closest[h][0]
            sortieL[i][j] = sortieL[i][j] / np.sum(
                [five_closest[h][0] for h in range(n_interpolation)]
            )
    
    for i in range(n_sortie):
        for j in range(8):
            for h in range(n_interpolation):
                sortieH[i][j] += sortie_high[h][i][j] * five_closest[h][0]
            sortieH[i][j] = sortieH[i][j] / np.sum(
                [five_closest[h][0] for h in range(n_interpolation)]
            )
            
    if maree : return coef_maree*sortieL + (1-coef_maree)*sortieH
    return sortieL, sortieH


a = OS2NS(0.26, 3, 280, 40, .5, True)
print(len(a))