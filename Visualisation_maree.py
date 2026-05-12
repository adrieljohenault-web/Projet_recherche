import os
import matplotlib.pyplot as plt
from variables_globales import *
from datetime import datetime
from Fonction_de_transfert import nombre_fichier_sortie

debut = 0
fin = 240

def visu_dpt_large():
    with open(
        os.path.join(path, "Vagues_forcage", "Tide_Brignogan_2009-2020_UTC_hourly.txt"), 'r'
    ) as file:
        lines = file.readlines()[1:]
        for i in range(len(lines)):
            lines[i] = lines[i].split()
        
    tide = [float(lines[i][5]) for i in range(debut, fin)]

    plt.plot(tide)
    plt.show()

def h_SL(k:int):
    LSL = []
    ch = os.path.join(
        path,
        "Delft3D_sorties_gamma04",
        "SL",
        f"D3D_res{nombre_fichier_sortie(k)}_SL.txt",
    )
    with open(ch, "r") as file:
        LSL = file.readlines()[1:]
    for j in range(len(LSL)):
        LSL[j] = LSL[j].split()
        LSL[j] = [float(x) for x in LSL[j]]
    SL = np.array(LSL)
    SL = SL[:,0]

    bathynp = np.array(bathy)
    bathynp = bathynp[:,0:2]


    data = np.column_stack((bathynp, SL))
    return data 


def visu_h_SL_blanc_zero(data, sondes=None):
    # 1. Conversion et nettoyage
    data = np.array(data, dtype=float)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # Filtrage des valeurs aberrantes (-999) sur les coordonnées ET la hauteur
    # On garde les points où X et Y sont valides
    mask_valid = (x > -500) & (y > -500)
    x_p, y_p, z_p = x[mask_valid], y[mask_valid], z[mask_valid]

    # 2. Création des masques pour le rendu
    mask_zero = (z_p <= 1e-5)  # Proche de 0 (gestion des arrondis flottants)
    mask_pos = (z_p > 1e-5)    # Strictement positif


    plt.figure(figsize=(12, 10))
    
    # --- ÉTAPE A : Affichage des points à 0 en BLANC ---
    # On peut ajouter un petit 'edgecolors' gris très clair si on veut les deviner sur fond blanc
    plt.scatter(x_p[mask_zero], y_p[mask_zero], color='white', s=33, label="Zone sèche")

    # --- ÉTAPE B : Affichage des points avec de l'eau ---
    if np.any(mask_pos):
        sc = plt.scatter(x_p[mask_pos], y_p[mask_pos], c=z_p[mask_pos], 
                        cmap='viridis', s=33, edgecolors='none')
        
        # On ajoute la barre de couleur uniquement pour les valeurs positives
        cbar = plt.colorbar(sc)
        cbar.set_label('Hauteur d\'eau (m)', rotation=270, labelpad=15)
    else:
        print("Note : Aucune hauteur d'eau positive détectée sur cette frame.")

    # 3. Cosmétique (Sondes et axes)
    if sondes:
        plt.scatter([s[0] for s in sondes], [s[1] for s in sondes], 
                    c='red', marker='+', s=200, label='Sondes')

    plt.title("Niveau de la marée - Zone du Vougot (0m = Blanc)", fontsize=18)
    plt.xlabel("Coordonnées X")
    plt.ylabel("Coordonnées Y")
    
    # On ajuste les limites sur les données valides
    plt.xlim(x_p.min() - 100, x_p.max() + 100)
    plt.ylim(y_p.min() - 100, y_p.max() + 100)
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()

visu_h_SL_blanc_zero(h_SL(1200), [sonde1,sonde2,sonde3])

def get_h_sup_0():
    L_H_sup_0 = []
    for i in range(1, n_valeurs_calc+1):
        data = h_SL(i)
        h = 0
        for pt, w in points_and_weights[0]:
            h += data[pt][2]

        if h>0.001:
            L_H_sup_0.append(i)
        print(i/n_valeurs_calc)
    
    return L_H_sup_0

#print(get_h_sup_0())
# --> [], donc aucune valeurs dans les fichiers de SL, tel que h soit non nul 


