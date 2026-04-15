import matplotlib.pyplot as plt
import numpy as np

n_sortie = 36225

sonde1 = [393153.67, 5388301.62]
sonde2 = [392928.35, 5388515.85]
sonde3 = [392450.79, 5389091.85]

sondes = [sonde1, sonde2, sonde3]

with open(
    "Fichier_d_encadrement/Lookup_table_Vougot/Delft3D_bathy/G2_Depth.txt",
    "r",
) as file:
    lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

entree = [[float(lines[i][j]) for j in range(2)] for i in range(n_sortie)]


def distance_euclidienne(q1: list, q2: list):
    """Calcule la distance euclidienne entre les deux array de même dimension q1 et q2"""
    distcarre = 0
    for k in range(len(q1)):
        distcarre += (q1[k] - q2[k]) ** 2
    return np.sqrt(distcarre)

Lsonde1 = []
Lsonde2 = []
Lsonde3 = []

for k in range(len(entree)):
    d1 = distance_euclidienne(entree[k], sonde1)
    d2 = distance_euclidienne(entree[k], sonde2)
    d3 = distance_euclidienne(entree[k], sonde3)
    Lsonde1.append(d1)
    Lsonde2.append(d2)
    Lsonde3.append(d3)

Lsonde1.sort()
Lsonde2.sort()
Lsonde3.sort()


points_1 = [Lsonde1[i] for i in range(4)]
points_2 = [Lsonde2[i] for i in range(4)]
points_3 = [Lsonde3[i] for i in range(4)]
print(points_1, points_2, points_3)