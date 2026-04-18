from Fonction_de_transfert import distance_euclidienne, n_sortie
from Schéma_bathymétrie import bathy, sondes
import numpy as np

### Choix des 4 points les plus proches des sondes pour réaliser une interpolation linéaire et choisir un point moyen qui les représentera. ###

# Création des poids pour l'interpolation

d_sonde = [[0. for _ in range(n_sortie)] for _ in range(3)]

for k in range(n_sortie):
    for i in range(3):
        d_sonde[i][k] = distance_euclidienne([bathy[k][0], bathy[k][1]], sondes[i])

points_inter = [[[] for _ in range(4)] for _ in range(3)]

for i in range(4):
    for j in range(3):
        closest_index = np.argmin(d_sonde[j])
        closest_distance = np.min(d_sonde[j])
        closest_point = np.array([bathy[closest_index][0], bathy[closest_index][1]])
        points_inter[j][i] = [closest_point, closest_distance, closest_index]
        d_sonde[j][closest_index] = 10**10

print(points_inter)

# Interpolation

S = []  # S[i] donnera le point interpolé représentant la sonde i

for i in range(3):
    point = np.array([0., 0.])
    s = 0
    for j in range(4):
        point += points_inter[i][j][0]*points_inter[i][j][1]
        s += points_inter[i][j][1]
    point = point/s
    S.append(point)