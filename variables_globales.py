n_valeurs_calc = 1761  # nombre de valeur d'entrée et de sortie qui ont été calculées par Delft3D (=len(entree))
n_interpolation = 5  # nombre de cas regardés pour l'interpolation
n_sortie = 36225  # nombre de lignes d'un fichier de sortie

# ouvrir les 3 blocks et les assembler dans entree de Delft3D

# Il s'agit des chemins associés au dossier Lookup_table_Vougot
path_clem = "/Users/clementcotte-grubis/office/PRECH3/repo_PRECH3/Lookup_table_Vougot"
path_adriel = "/Users/adrielhenault/Documents/ECOLE DES PONTS/1A/TRAVAIL/SCIENTIFIQUE/PARCOURS RECHERCHE/Fichier_d_encadrement/Lookup_table_Vougot"

sonde1 = [393153.67, 5388301.62]
sonde2 = [392928.35, 5388515.85]
sonde3 = [392450.79, 5389091.85]

sondes = [sonde1, sonde2, sonde3]
