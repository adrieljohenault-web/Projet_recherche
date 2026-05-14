# Le résonnement est le suivant. Si on suppose que Delft3D dit la vérité,
# c'est que le problème vient de comment on calcul le niveau de la marée. 


# le but est de construire ^y = a * Sh + b * Sl, avec a,b > 0 des vecteurs de dimenssion 2

# imports

import os 
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import csv
from scipy.optimize import nnls

from variables_globales import *
from Fonction_de_transfert import *
from sonde_donnee_formatage import v1, v2, v3, vin,vin_sync



#### Étape 1 ####
# L'objectif ici est de construire les vecteurs d'apprentissage X. 
# définiton des entrées X
# X = [Hs_large, Tp_large, Dir_large, marnage, depth]
# Hs_large : la valeur de Hs au large
# Tp_large : la valeur de Tp au large
# Dir_large : la valeur de la direction au large
# marnage : la valeur entre marée haute et marée basse. On a observer que toutes les marées hautes n'avait pas la même hauteurs d'eau 
# depth : la profondeur moyenne du point de la sonde

# la taille de X est 7440 * 3, 7440 : nombre de points de données pour les 3 sondes, 3 : nombre de sondes
# X = [sonde1, sonde2, sonde3]

# calcul des depth

def get_mean_depth(v_sonde):
    depth = v_sonde[:,3]
    return np.mean(depth)

depth_v1 = get_mean_depth(v1)
depth_v2 = get_mean_depth(v2)
depth_v3 = get_mean_depth(v3)

# calcul du marnage 

def compute_marnage(v_large, target_dates, window_days=2):
    source_dates = v_large[:, 0]
    dpts = v_large[:, 4].astype(float)
    
    marnage_source = []
    for i, date in enumerate(source_dates):
        cutoff = date - timedelta(days=window_days)
        mask = (source_dates >= cutoff) & (source_dates <= date)
        
        if np.any(mask):
            val = dpts[mask].max() - dpts[mask].min()
            marnage_source.append(val)
        else:
            marnage_source.append(np.nan)
    
    marnage_source = np.array(marnage_source)

    indices = np.searchsorted(source_dates, target_dates)
    marnage_interp = []

    for i, date in enumerate(target_dates):
        k = indices[i]

        if k == 0 or k >= len(v_large):
            marnage_interp.append(np.nan)
            continue

        dt_inf = source_dates[k-1]
        dt_sup = source_dates[k]
        
        delta = (dt_sup - dt_inf).total_seconds()
        w_sup = (date - dt_inf).total_seconds() / delta
        w_inf = 1 - w_sup

        m_val = w_inf * marnage_source[k-1] + w_sup * marnage_source[k]
        marnage_interp.append(m_val)
    
    # Retourne un array avec [Date, Valeur_Marnage]
    return np.column_stack((target_dates, marnage_interp))

marnage = compute_marnage(vin, vin_sync[:, 0])



# création du vecteur X

def get_X(vin_sync, marn_data, depth1, depth2, depth3):
    mask = (vin_sync[:, 0] == marn_data[:, 0])
    # On ne garde que les lignes valides
    vin_sync_clean = vin_sync[mask]
    marn_val_clean = marn_data[mask, 1]
    
    # Optionnel : Alerte si des dates ont été supprimées
    diff = len(vin_sync) - len(vin_sync_clean)
    if diff > 0:
        print(f"Attention : {diff} lignes supprimées car les dates ne coïncidaient pas.")

    # 2. Construction du bloc commun : [Date, Hs, Tp, Dir, Marnage]
    # On utilise column_stack pour fusionner les colonnes proprement
    common = np.column_stack((vin_sync_clean, marn_val_clean))

    # 3. Ajout de la profondeur pour chaque sonde
    n = len(common)
    X1 = np.column_stack((common, np.full(n, depth1)))
    X2 = np.column_stack((common, np.full(n, depth2)))
    X3 = np.column_stack((common, np.full(n, depth3)) )

    # 4. Empilement vertical final
    return np.vstack((X1, X2, X3))

# Utilisation
X = get_X(vin_sync, marnage, depth_v1, depth_v2, depth_v3)


#### Étape 2 ####
# L'objectif est de construire les y = (a*,b*) désirée. cad, qu'on cherche a*,b* qui minimisent la norme de a*(SH,i) + b*(SL,i) - y,i pour toutes les dates i
# pour lesquelles on a des données des sondes

# coef = [a,b]      a: S_H, b:S_L

def get_normalization(v_sonde):
    mu_Hs = np.mean(v_sonde[:,1])
    sigma_Hs = np.std(v_sonde[:,1])

    mu_Tp = np.mean(v_sonde[:,2])
    sigma_Tp = np.std(v_sonde[:,2])

    return mu_Hs, sigma_Hs, mu_Tp, sigma_Tp

def verif_coeff(coeff, v_sonde):
    func = OS2NS_uni_pluriel(points_and_weights[0], vin_sync[0][1], vin_sync[0][2], vin_sync[0][3], 1, False)
    res = coeff[0]*func[1] + coeff[1]*func[0]

    print(res)
    print(v_sonde[0])

verif_coeff([0.79, 0],v1)

def get_hyp_ab(v_sonde, points, vin_sync):

    mu_Hs, sigma_Hs, mu_Tp, sigma_Tp = get_normalization(v_sonde)


    n_points = v_sonde.shape[0]
    arr_coefs = np.zeros((n_points, 3), dtype=object)

    for i in range(n_points):
        if v_sonde[i, 0] != vin_sync[i, 0]:
            print(f"Désynchronisation détectée à l'index {i}")
            continue

        y_i = (v_sonde[i, 1:3].astype(float)

        Hs, Tp, Dir = vin_sync[i, 1], vin_sync[i, 2], vin_sync[i, 3]
        pred_low, pred_high = OS2NS_uni_pluriel(points, Hs, Tp, Dir, 0, False)

        S_H_i = pred_high[:, :2].astype(float).flatten()
        S_L_i = pred_low[:, :2].astype(float).flatten()

        A = np.column_stack([S_H_i, S_L_i])


        coeffs, _ = nnls(A, y_i)

        arr_coefs[i, 0] = v_sonde[i, 0]
        arr_coefs[i, 1] = coeffs[0]    # Coefficient a (S_H)
        arr_coefs[i, 2] = coeffs[1]    # Coefficient b (S_L)

        print(coeffs[0], coeffs[1])
    
    return arr_coefs



#y_resultat = get_hyp_ab(v1, points_and_weights[0], vin_sync)


"""
Remplacement optimisé de get_hyp_ab (IA2.py).

À coller dans IA2.py à la place de get_hyp_ab, puis remplacer l'appel :
    y_resultats = get_hyp_ab_fast(v1, points_and_weights[0], vin_sync)

Optimisations par rapport à l'original
───────────────────────────────────────
1. Calcul vectorisé (batch) des distances pour TOUS les pas de temps d'un coup,
   au lieu de recalculer la boucle n_valeurs_calc dans OS2NS_uni à chaque itération.
   → Réduction O(n_pts × n_valeurs_calc) boucle Python → une seule opération numpy.

2. Cache dict de sortie_fichier_uni(file_idx, grid_point) : la même paire peut
   revenir des centaines de fois (n_pts >> n_valeurs_calc), le fichier n'est lu
   qu'une seule fois.
   → Préchargement de toutes les paires uniques nécessaires avant la boucle principale.

3. Correction du double-appel à OS2NS_uni dans OS2NS_uni_pluriel :
   l'original appelait OS2NS_uni(...)[0] PUIS OS2NS_uni(...)[1] séparément,
   ce qui doublait inutilement les lectures disque.

4. Calcul chunké des distances pour limiter la consommation mémoire de pointe
   (paramètre chunk_size, défaut 500 pas de temps à la fois).
"""

import numpy as np
from scipy.optimize import nnls

# ── Les variables _means, _entree_norm, n_interpolation, sortie_fichier_uni
#    sont importées depuis Fonction_de_transfert via le wildcard import existant.


def get_hyp_ab_fast(v_sonde, points, vin_sync, chunk_size: int = 500):
    """Version optimisée de get_hyp_ab.

    Paramètres
    ----------
    v_sonde     : array (n_pts, ≥3)  – données sonde [date, Hs, Tp, ...]
    points      : list[(grid_pt, weight)]  – points de grille G2 et leurs poids
    vin_sync    : array (n_pts, ≥4)  – conditions au large synchronisées
                  [date, Hs, Tp, Dir, ...]
    chunk_size  : int – nombre de pas de temps traités à la fois pour le calcul
                  des distances (réduit la mémoire de pointe, défaut 500)

    Retour
    ------
    arr_coefs : array (n_pts, 3) dtype=object – [date, coef_a (S_H), coef_b (S_L)]
    """

    n_pts = v_sonde.shape[0]

    # ── Étape 1 : calcul batch des distances ──────────────────────────────────
    #
    # inputs    : (n_pts, 3)         – Hs, Tp, Dir normalisés
    # distances : (n_pts, n_calc)    – distance euclidienne à chaque condition de lookup table
    #
    inputs    = vin_sync[:n_pts, 1:4].astype(float)
    args_norm = inputs / means_                          # (n_pts, 3)

    n_calc    = entree_norm_.shape[0]
    distances = np.empty((n_pts, n_calc), dtype=np.float64)

    for start in range(0, n_pts, chunk_size):
        end  = min(start + chunk_size, n_pts)
        diff = args_norm[start:end, np.newaxis, :] - entree_norm_[np.newaxis, :, :]
        # diff : (chunk, n_calc, 3)  →  distances : (chunk, n_calc)
        distances[start:end] = np.sqrt(np.sum(diff ** 2, axis=2))

    # n_interpolation plus proches voisins pour chaque pas de temps
    indices_all = np.argsort(distances, axis=1)[:, :n_interpolation]       # (n_pts, k)
    dist_all    = np.take_along_axis(distances, indices_all, axis=1)        # (n_pts, k)

    poids_all   = 1.0 / (dist_all + 1e-12)
    poids_all  /= poids_all.sum(axis=1, keepdims=True)                      # normalisés

    # ── Étape 2 : préchargement du cache de fichiers ──────────────────────────
    #
    # On identifie toutes les paires (file_idx, grid_point) utiles AVANT la boucle
    # et on les lit une seule fois. La même paire peut être demandée pour des
    # centaines de pas de temps différents → gain majeur sur les I/O.
    #
    grid_pts = [int(gpt) for gpt, _ in points]
    grid_ws  = np.array([w for _, w in points], dtype=float)   # (n_gpts,)

    unique_pairs = {
        (int(indices_all[i, h]), gpt)
        for i in range(n_pts)
        for h in range(n_interpolation)
        for gpt in grid_pts
    }

    print(f"[get_hyp_ab_fast] Préchargement de {len(unique_pairs)} paires (fichier, point)…")
    _file_cache: dict = {}
    for file_idx, gpt in unique_pairs:
        _file_cache[(file_idx, gpt)] = sortie_fichier_uni(file_idx, gpt)
    print("[get_hyp_ab_fast] Préchargement terminé.")

    # ── Étape 3 : boucle principale ───────────────────────────────────────────
    arr_coefs = np.zeros((n_pts, 3), dtype=object)

    for i in range(n_pts):
        if v_sonde[i, 0] != vin_sync[i, 0]:
            print(f"Désynchronisation à l'index {i}")
            continue

        y_i           = v_sonde[i, 1:3].astype(float)
        neighbor_idxs = indices_all[i]   # (k,) indices dans la lookup table
        poids         = poids_all[i]     # (k,) poids IDW normalisés

        # Accumulation pondérée (IDW) de sortie_fichier_uni sur les k voisins,
        # pour chaque point de grille → shape (n_gpts, 8)
        acc_L = np.zeros((len(grid_pts), 8))
        acc_H = np.zeros((len(grid_pts), 8))

        for h, file_idx in enumerate(neighbor_idxs):
            for p, gpt in enumerate(grid_pts):
                lsl, lsh = _file_cache[(int(file_idx), gpt)]   # lecture depuis le cache
                acc_L[p] += lsl * poids[h]
                acc_H[p] += lsh * poids[h]

        # Somme pondérée sur les points de grille → vecteur (8,)
        S_L = (acc_L * grid_ws[:, np.newaxis]).sum(axis=0)
        S_H = (acc_H * grid_ws[:, np.newaxis]).sum(axis=0)

        # Résolution nnls : min_a,b ||a*S_H[:2] + b*S_L[:2] - y_i||
        # (identique à l'original, système 2×2)
        A = np.column_stack([S_H[:2], S_L[:2]])
        coeffs, _ = nnls(A, y_i)

        arr_coefs[i, 0] = v_sonde[i, 0]
        arr_coefs[i, 1] = coeffs[0]    # coefficient a (S_H)
        arr_coefs[i, 2] = coeffs[1]    # coefficient b (S_L)

        if i % 200 == 0:
            print(f"  {i}/{n_pts}  ({100*i/n_pts:.0f}%)")

    return arr_coefs


# ── Utilisation (remplace la ligne existante dans IA2.py) ─────────────────────
#
#y_resultats = get_hyp_ab_fast(v1, points_and_weights[0], vin_sync)

#print(y_resultats)

