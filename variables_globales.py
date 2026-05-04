import numpy as np
import os

n_valeurs_calc = 1761  # nombre de valeur d'entrée et de sortie qui ont été calculées par Delft3D (=len(entree))
n_interpolation = 5  # nombre de cas regardés pour l'interpolation
n_sortie = 36225  # nombre de lignes d'un fichier de sortie

# ouvrir les 3 blocks et les assembler dans entree de Delft3D

# Il s'agit des chemins associés au dossier Lookup_table_Vougot
path_clem = "/Users/clementcotte-grubis/office/PRECH3/repo_PRECH3/Lookup_table_Vougot"
path_adriel = "/Users/adrielhenault/Documents/ECOLE DES PONTS/1A/TRAVAIL/SCIENTIFIQUE/PARCOURS RECHERCHE/Fichier_d_encadrement/Lookup_table_Vougot"
path = path_adriel

# Emplacements des sondes

sonde1 = [393153.67, 5388301.62]
sonde2 = [392928.35, 5388515.85]
sonde3 = [392450.79, 5389091.85]

sondes = [sonde1, sonde2, sonde3]

def distance_euclidienne(q1: list, q2: list):
    """Calcule la distance euclidienne entre les deux array de même dimension q1 et q2"""
    distcarre = 0
    for k in range(len(q1)):
        distcarre += (q1[k] - q2[k]) ** 2
    return np.sqrt(distcarre)

# Bathymétrie de la zone G2

with open(
    os.path.join(path, "Delft3D_bathy", "G2_Depth.txt"),
    "r",
) as file:
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

bathy = [[float(lines[i][j]) for j in range(3)] for i in range(n_sortie)]

# 4 points les plus proches de chaque sonde et leur distance à la sonde

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

# Somme des distances des 4 points pour chaque sonde

somme_dist = [sum([points_inter[i][j][1] for j in range(4)]) for i in range(3)]
inv_sum_dist = [sum([1/points_inter[i][j][1] for j in range(4)]) for i in range(3)]

#calcul des poids associés à la distance 
points_and_weights = [
    [[points_inter[j][i][0], (1/points_inter[j][i][1]) / inv_sum_dist[j]] 
     for i in range(4)] 
    for j in range(3)
]