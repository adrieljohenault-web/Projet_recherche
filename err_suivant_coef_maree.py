"""
Analyse de l'erreur de la fonction de transfert en fonction du coefficient de marée.

Ce module est conçu pour être importé depuis verif_v2.py (ou lancé après que v_marree,
v1, v2, v3 et le CSV d'erreurs ont été produits). Il aligne les erreurs et le coef de
marée par date, puis trace un histogramme de l'erreur moyenne par bin de coef.

Usage typique (depuis verif_v2.py, après le calcul de v_marree) :

    from err_suivant_coef_maree import plot_err_vs_coef_maree
    plot_err_vs_coef_maree(v_marree, nom_fichier="erreurs_all_transfer_function.csv")
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime


def _lire_csv_erreurs(nom_fichier: str):
    """
    Lit le CSV d'erreurs et retourne :
      - dates_err  : list[datetime]
      - data_err   : np.ndarray, shape (N, 6)  — colonnes Hs1,Tp1,Hs2,Tp2,Hs3,Tp3
    """
    dates_err = []
    rows = []

    with open(nom_fichier, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)                                # saute le header
        for row in reader:
            dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            dates_err.append(dt)
            rows.append([float(v) for v in row[1:]])

    return dates_err, np.array(rows)


def plot_err_vs_coef_maree(
    v_marree: np.ndarray,
    nom_fichier: str = "erreurs_all_transfer_function.csv",
    n_bins: int = 10,
    save_path: str = None,
):
    """
    Trace 6 histogrammes (un par variable : Hs_s1, Tp_s1, Hs_s2, Tp_s2, Hs_s3, Tp_s3)
    montrant l'erreur absolue moyenne en fonction du coef de marée.

    Paramètres
    ----------
    v_marree    : array (N, 2) — colonnes [datetime, coef]  (issu de get_coef_marree)
    nom_fichier : chemin vers le CSV d'erreurs
    n_bins      : nombre de bins de coef de marée (défaut : 10)
    save_path   : si fourni, sauvegarde la figure à ce chemin
    """

    # ------------------------------------------------------------------
    # 1. Lecture du CSV
    # ------------------------------------------------------------------
    dates_err, data_err = _lire_csv_erreurs(nom_fichier)
    dates_err_arr = np.array(dates_err, dtype=object)

    # ------------------------------------------------------------------
    # 2. Alignement par date (intersection)
    # ------------------------------------------------------------------
    dates_marree = set(v_marree[:, 0])
    dates_csv    = set(dates_err_arr)
    dates_communes = sorted(dates_marree & dates_csv)

    if len(dates_communes) == 0:
        raise ValueError(
            "Aucune date commune entre v_marree et le CSV. "
            "Vérifiez que les deux proviennent du même pipeline synchronisé."
        )

    # Index dans chaque tableau
    dates_marree_arr = v_marree[:, 0]
    date_to_coef = {dt: float(coef) for dt, coef in v_marree}
    date_to_err  = {dt: data_err[i] for i, dt in enumerate(dates_err_arr)}

    coefs  = np.array([date_to_coef[dt] for dt in dates_communes])
    errors = np.array([date_to_err[dt]  for dt in dates_communes])
    # errors : shape (M, 6)

    # ------------------------------------------------------------------
    # 3. Binning du coef de marée
    # ------------------------------------------------------------------
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_labels = [f"{bin_edges[k]:.1f}–{bin_edges[k+1]:.1f}" for k in range(n_bins)]
    bin_idx    = np.digitize(coefs, bin_edges, right=False) - 1
    bin_idx    = np.clip(bin_idx, 0, n_bins - 1)   # protection bords

    # Moyenne des erreurs dans chaque bin (NaN si bin vide)
    means = np.full((n_bins, 6), np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = errors[mask].mean(axis=0)

    # ------------------------------------------------------------------
    # 4. Tracé
    # ------------------------------------------------------------------
    var_names  = ["Hs_s1", "Tp_s1", "Hs_s2", "Tp_s2", "Hs_s3", "Tp_s3"]
    var_units  = ["m",     "s",     "m",     "s",     "m",     "s"]
    colors     = ["#2196F3", "#64B5F6",   # bleus pour s1
                  "#E53935", "#EF9A9A",   # rouges pour s2
                  "#43A047", "#A5D6A7"]   # verts pour s3

    x = np.arange(n_bins)
    bar_width = 0.65

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    fig.suptitle(
        "Erreur absolue moyenne de la fonction de transfert\nen fonction du coefficient de marée",
        fontsize=14, fontweight="bold", y=1.01
    )

    for idx, ax in enumerate(axes.flat):
        vals = means[:, idx]
        bars = ax.bar(
            x, vals,
            width=bar_width,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        # Annotation du nombre de points par bin
        for b, bar in enumerate(bars):
            if counts[b] > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * np.nanmax(vals),
                    f"n={counts[b]}",
                    ha="center", va="bottom",
                    fontsize=7, color="#555555"
                )

        ax.set_title(var_names[idx], fontsize=12, fontweight="semibold")
        ax.set_ylabel(f"Erreur absolue moyenne ({var_units[idx]})", fontsize=9)
        ax.set_xlabel("Coefficient de marée", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ligne de la moyenne globale
        global_mean = np.nanmean(vals)
        ax.axhline(global_mean, color="black", linestyle="--", linewidth=1.0,
                   label=f"Moy. globale : {global_mean:.4f} {var_units[idx]}")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")

    plt.show()
    return fig, axes


# ------------------------------------------------------------------
# Optionnel : lancement standalone (reconstruit v_marree depuis les données brutes)
# ------------------------------------------------------------------
if False:
    import os
    import sys

    # Ajouter le répertoire parent au path pour accéder aux modules du projet
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from variables_globales import path
    from verif_v2 import format_sonde, get_coef_marree, synchroniser

    with open(os.path.join(path, "Capteurs_pression",
              "S3_capteur_offshore_2012-12-11_2013-03-14_waveStats_filt_h01.1.dat"), 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].split()
    verif_3 = np.array(lines)[300:-311]
    v3 = format_sonde(verif_3)

    v_marree = get_coef_marree(v3)

    plot_err_vs_coef_maree(
        v_marree,
        nom_fichier="erreurs_all_transfer_function.csv",
        n_bins=10,
        save_path="err_vs_coef_maree.png",
    )