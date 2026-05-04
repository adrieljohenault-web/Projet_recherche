import numpy as np
import os
from variables_globales import *


with open(
    os.path.join(path, "Delft3D_entrees", "allblocks_Vougot.txt"),
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
        path,
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
        path,
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

# Quantités pour les normalisations

mean_Hs = np.average(np.array([entree[i][0] for i in range(n_valeurs_calc)]))
mean_Tp = np.average(np.array([entree[i][1] for i in range(n_valeurs_calc)]))
mean_Dir = np.average(np.array([entree[i][2] for i in range(n_valeurs_calc)]))

means = [mean_Hs, mean_Tp, mean_Dir]

# Normalisation de l'entrée

entree_norm = np.zeros((n_valeurs_calc, 3))  # dtype float64 par défaut
for i in range(n_valeurs_calc):
    for j in range(3):
        entree_norm[i][j] = entree[i][j]/means[j]

def OS2NS(Hs: float, Tp: float, Dir: float, coef_maree: float, maree: bool = True) -> list:
    """Fonction de transfert en tant que telle : réalise l'interpolation qui permet d'obtenir les condtitions de déferlement. Retourne la matrice des conditions de déferlement en chaque point.
    Le paramètre coef_maree est un coefficient entre 0 et 1 permettant de tenir compte de la dépendance linéaire des conditions de sortie en fonction de la maree.
    Si maree est réglé sur True, alors la marée sera prise en compte et une seule valeur correspondante sera retournée. Dans le cas contraire, la fonction retourne deux valeurs de sortie correspondant respectivement aux marées basse et haute."""

    # Parcourir l'ensemble des données d'entrées et prendre les 5 plus petites distances euclidiennes à la situation en argument

    arg = [Hs, Tp, Dir]

    # Normalisation de argument et de l'entrée pour application des poids à l'interpolation plus tard

    arg_norm = np.array([Hs/mean_Hs, Tp/mean_Tp, Dir/mean_Dir])
    
    five_closest = [[], [], [], [], []]

    memo_norm: list[float] = np.array([0. for _ in range(n_valeurs_calc)])
    for i in range(n_valeurs_calc):
        memo_norm[i] = distance_euclidienne(arg_norm, entree_norm[i])

    # Faire 5 minima en enlevant la valeur minimale à chaque fois ; stocker la distance et l'indice dans le tableau five_closest

    for i in range(n_interpolation):
        closest_index = int(np.argmin(memo_norm))
        closest_distance = float(np.min(memo_norm))
        closest_point = [float(entree[np.argmin(memo_norm)][i]) for i in range(4)]
        five_closest[i] = [closest_distance, closest_index, closest_point]
        memo_norm[np.argmin(memo_norm)] = 1000000

    # Arriver aux données de sortie de Delft3D

    # sortie_low = [[tableau des conditions suivantes en chaque point du maillage] pour chaque condition d'entrée des five_closest] --> Hs, Tp, Tm01, Dp, Dm, DSpr, WD, Qb
    # sortie_high = ...

    sortie_low, sortie_high = [
        sortie_fichier(five_closest[i][1])[0] for i in range(n_interpolation)
    ], [sortie_fichier(five_closest[i][1])[1] for i in range(n_interpolation)]

    # Faire l'interpolation et retourner les valeurs finales

    sortieL, sortieH = np.array([[0.] * 8 for _ in range(n_sortie)]), np.array([[0.] * 8 for _ in range(n_sortie)])

    for i in range(n_sortie):
        for j in range(8):
            for h in range(n_interpolation):
                sortieL[i][j] += sortie_low[h][i][j] * (1/(five_closest[h][0]+10**(-12)))
            sortieL[i][j] = sortieL[i][j] / np.sum(
                [1/(five_closest[h][0]+10**(-12)) for h in range(n_interpolation)]
            )
    
    for i in range(n_sortie):
        for j in range(8):
            for h in range(n_interpolation):
                sortieH[i][j] += sortie_high[h][i][j] * (1/(five_closest[h][0]+10**(-12)))
            sortieH[i][j] = sortieH[i][j] / np.sum(
                [1/(five_closest[h][0]+10**(-12)) for h in range(n_interpolation)]
            )
            
    if maree : return coef_maree*sortieL + (1-coef_maree)*sortieH
    return sortieL, sortieH

_means = np.mean(entree[:, :3], axis=0)
_entree_norm = entree[:, :3] / _means

def OS2NS_vectorized(Hs: float, Tp: float, Dir: float, coef_maree: float, maree: bool = True) -> list:
    """Fonction de transfert en tant que telle : réalise l'interpolation qui permet d'obtenir les condtitions de déferlement. Retourne la matrice des conditions de déferlement en chaque point.
    Le paramètre coef_maree est un coefficient entre 0 et 1 permettant de tenir compte de la dépendance linéaire des conditions de sortie en fonction de la maree.
    Si maree est réglé sur True, alors la marée sera prise en compte et une seule valeur correspondante sera retournée. Dans le cas contraire, la fonction retourne deux valeurs de sortie correspondant respectivement aux marées basse et haute."""

    # Parcourir l'ensemble des données d'entrées et prendre les 5 plus petites distances euclidiennes à la situation en argument
    arg = np.array([Hs, Tp, Dir])


    # Normalisation de argument et de l'entrée pour application des poids à l'interpolation plus tard
    arg_norm = arg / _means
    entree_norm = _entree_norm

    distances = np.sqrt(np.sum((entree_norm - arg_norm)**2, axis=1))

    indices_proches = np.argsort(distances)[:n_interpolation]
    dist_proches = distances[indices_proches]

    poids = 1.0 / (dist_proches+10**(-12)) # On ajoute un micro-chiffre pour éviter la division par 0

    sortie_low = np.array([sortie_fichier(idx)[0] for idx in indices_proches])
    sortie_high = np.array([sortie_fichier(idx)[1] for idx in indices_proches])

    sortieL = np.average(sortie_low, axis=0, weights=poids)
    sortieH = np.average(sortie_high, axis=0, weights=poids)
            
    if maree: 
        return coef_maree * sortieL + (1 - coef_maree) * sortieH
    return sortieL, sortieH

def OS2NS_vectorized_per_points(Hs: float, Tp: float, Dir: float, coef_maree: float, 
                                 list_points: list, maree: bool = True) -> list:
    arg = np.array([Hs, Tp, Dir])


    arg_norm = arg / _means
    entree_norm = _entree_norm

    distances = np.sqrt(np.sum((entree_norm - arg_norm)**2, axis=1))
    indices_proches = np.argsort(distances)[:n_interpolation]
    dist_proches = distances[indices_proches]
    poids = 1.0 / (dist_proches + 1e-12)

    sortie_low = np.array([sortie_fichier(idx)[0][list_points] for idx in indices_proches])
    sortie_high = np.array([sortie_fichier(idx)[1][list_points] for idx in indices_proches])

    sortieL = np.average(sortie_low, axis=0, weights=poids)
    sortieH = np.average(sortie_high, axis=0, weights=poids)

    if maree:
        return coef_maree * sortieL + (1 - coef_maree) * sortieH
    return sortieL, sortieH


#fonctino de verification
def verif_transfert_funct():
    a = OS2NS(0.25, 22.5, 270, 0.5)
    b = OS2NS_vectorized(0.25, 22.5, 270, 0.5)
    eps = 0.01           # précision que l'on veut avoir 
    for i in range(len(a)):
        show = False
        for j in range(len(a[0])):
            if not (b[i][j]*(1-eps) <= a[i][j] <= b[i][j]*(1+eps)):
                show = True 
        if show:
            print(a[i], b[i])


def verif_transfert_funct_per_points():
    points = [17000,15000,11000,9000]
    a = OS2NS_vectorized_per_points(0.25, 22.5, 270, 0.5, points)
    b = OS2NS_vectorized(0.25, 22.5, 270, 0.5)
    eps = 0.01           # précision que l'on veut avoir 
    for i in range(len(points)):
        for j in range(len(a[0])):
            if not (b[points[i]][j]*(1-eps) <= a[i][j] <= b[points[i]][j]*(1+eps)):
                print("aie")
            print(a[i], b[points[i]])


def OS2NS_uni(point: int, Hs: float, Tp: float, Dir: float, coef_maree: float, maree: bool = True):
    """Fonction de transfert optimisée qui renvoie un résultat en un seul point."""
