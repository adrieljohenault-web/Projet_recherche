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

entree = [[float(lines[i][j]) for j in range(3)] for i in range(n_sortie)]

import numpy as np
import matplotlib.pyplot as plt

def generer_carte_bathymetrique_recherche(data_list):
    """
    Représentation 2D stricte des points de données.
    Aucune interpolation, aucune création de points inexistants.
    """
    # 1. Validation de la structure d'entrée
    if not data_list or not isinstance(data_list, (list, np.ndarray)):
        print("Erreur : La liste d'entrée est vide ou invalide.")
        return

    # 2. Conversion forcée et gestion du typage (pour gérer le format scientifique)
    try:
        # On convertit tout en float. Si une ligne est corrompue, elle est ignorée.
        data = np.array(data_list, dtype=float)
    except Exception as e:
        print(f"Erreur lors de la conversion des données : {e}")
        return

    # 3. Extraction des colonnes [x, y, z]
    try:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
    except IndexError:
        print("Erreur : Les sous-listes doivent contenir au moins 3 éléments [x, y, z].")
        return

    # 4. Filtrage strict des NaNs (-999)
    # On exclut -999 et on vérifie aussi les éventuels NaN réels (np.nan)
    mask = (z != -999) & (~np.isnan(z))
    
    x_pure = x[mask]
    y_pure = y[mask]
    z_pure = z[mask]

    for i in range(len(z_pure)):
        z_pure[i] = -z_pure[i]

    M = max(z_pure)
    for i in range(len(z_pure)):
        z_pure[i] -= M

    if len(z_pure) == 0:
        print("Alerte : Aucun point valide après filtrage des -999.")
        return

    # 5. Visualisation 2D fidèle (Heatmap par points)
    plt.figure(figsize=(12, 10))
    
    # On utilise scatter : chaque point est une donnée RÉELLE du dataset
    # 's' définit la taille du point. 'edgecolors=none' évite les artefacts visuels.
    sc = plt.scatter(x_pure, y_pure, c=z_pure, cmap='viridis', s=33, edgecolors='none')
    plt.scatter([sondes[i][0] for i in range(3)], [sondes[i][1] for i in range(3)], c = 'black', marker = '+', s = 200)
    # Ajout d'une barre de couleur avec label scientifique
    cbar = plt.colorbar(sc)
    cbar.set_label('Profondeur (m)', rotation=270, labelpad=15)

    # Configuration des axes
    plt.title(f"Bathymétrie de la section 2 de la plage du Vougot", fontsize=24)
    plt.xlabel("Coordonnées X")
    plt.ylabel("Coordonnées Y")
    
    plt.xlim(39600,384500)
    plt.ylim(5386000, 5394000)
    # Aspect ratio 1:1 pour ne pas déformer les distances réelles
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- Exécution ---
generer_carte_bathymetrique_recherche(entree)