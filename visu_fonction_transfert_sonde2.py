"""
Script de visualisation des résultats de la fonction de transfert pure pour la sonde 2.
Calcul uniquement sur les 5 premiers jours avec la fonction OS2NS_uni_pluriel.
"""

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Imports des modules du projet
from variables_globales import *
from Fonction_de_transfert import *
from sonde_donnee_formatage import v1, v2, v3, vin_sync, v_marree


# =============================================================================
# FONCTIONS DE VISUALISATION (Identiques au format IA2)
# =============================================================================

def plot_comparison(dates, real_Hs, real_Tp, pred_Hs, pred_Tp,
                    title="Sonde 2 : Comparaison Données Réelles vs Fonction de Transfert (5 jours)",
                    save_path="comparison_sonde2_5days_transfer.png"):
    """Crée un graphique comparatif Hs et Tp en fonction du temps."""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # =========================================================================
    # Graphique Hs
    # =========================================================================
    ax1 = axes[0]

    # Données réelles : points uniquement
    ax1.scatter(
        dates,
        real_Hs,
        color='blue',
        label='Hs Réel (sonde)',
        alpha=0.7,
        s=20
    )

    # Fonction de transfert : ligne + points
    ax1.plot(
        dates,
        pred_Hs,
        'r--',
        alpha=0.8,
        linewidth=1.5,
        label='Hs Fonction de Transfert'
    )

    ax1.scatter(
        dates,
        pred_Hs,
        color='red',
        alpha=0.7,
        s=15
    )

    ax1.set_ylabel('Hs [m]', fontsize=12)
    ax1.set_title(f'{title} - Hauteur Significative',
                  fontsize=14,
                  fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Graphique Tp
    # =========================================================================
    ax2 = axes[1]

    # Données réelles : points uniquement
    ax2.scatter(
        dates,
        real_Tp,
        color='blue',
        label='Tp Réel (sonde)',
        alpha=0.7,
        s=20
    )

    # Fonction de transfert : ligne + points
    ax2.plot(
        dates,
        pred_Tp,
        'r--',
        alpha=0.8,
        linewidth=1.5,
        label='Tp Fonction de Transfert'
    )

    ax2.scatter(
        dates,
        pred_Tp,
        color='red',
        alpha=0.7,
        s=15
    )

    ax2.set_ylabel('Tp [s]', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'{title} - Période de Pic',
                  fontsize=14,
                  fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Graphique temporel sauvegardé : {save_path}")

    plt.show()


def plot_scatter_comparison(real_Hs, real_Tp, pred_Hs, pred_Tp,
                            save_path="scatter_sonde2_5days_transfer.png"):
    """Crée des scatter plots réel vs prédit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter Hs
    ax1 = axes[0]
    ax1.scatter(real_Hs, pred_Hs, alpha=0.5, c='blue', s=20)
    max_hs = max(max(real_Hs), max(pred_Hs)) if len(real_Hs) > 0 else 1
    ax1.plot([0, max_hs], [0, max_hs], 'r--', lw=2, label='y=x (parfait)')
    ax1.set_xlabel('Hs Réel [m]', fontsize=12)
    ax1.set_ylabel('Hs Prédit [m]', fontsize=12)
    ax1.set_title('Hs : Réel vs Prédit', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter Tp
    ax2 = axes[1]
    ax2.scatter(real_Tp, pred_Tp, alpha=0.5, c='green', s=20)
    max_tp = max(max(real_Tp), max(pred_Tp)) if len(real_Tp) > 0 else 1
    ax2.plot([0, max_tp], [0, max_tp], 'r--', lw=2, label='y=x (parfait)')
    ax2.set_xlabel('Tp Réel [s]', fontsize=12)
    ax2.set_ylabel('Tp Prédit [s]', fontsize=12)
    ax2.set_title('Tp : Réel vs Prédit', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot sauvegardé : {save_path}")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("VISUALISATION FONCTION DE TRANSFERT - SONDE 2 (5 PREMIERS JOURS)")
    print("=" * 70)

    # 1. Détermination de la fenêtre temporelle : 5 premiers jours
    first_date = min(v2[:, 0])
    last_date = first_date + datetime.timedelta(days=5)

    print(f"   → Première date sonde 2 : {first_date}")
    print(f"   → Fenêtre 5 jours       : {first_date} à {last_date}")

    # 2. Filtrage des données (déjà synchronisées dans sonde_donnee_formatage)
    mask_5days = np.array([first_date <= d <= last_date for d in v2[:, 0]])
    
    v2_5days = v2[mask_5days]
    vin_sync_5days = vin_sync[mask_5days]
    v_marree_5days = v_marree[mask_5days]

    real_dates = v2_5days[:, 0]
    real_Hs = v2_5days[:, 1].astype(float)
    real_Tp = v2_5days[:, 2].astype(float)

    # 3. Reconstruction via la fonction de transfert pure
    print("\n[3/5] Calcul des prédictions de la fonction de transfert...")
    
    pred_Hs_list = []
    pred_Tp_list = []
    points = points_and_weights[1]  # Extraction des points et poids de la sonde 2

    for i in range(len(v2_5days)):
        coef = float(v_marree_5days[i][1])
        Hs_large = float(vin_sync_5days[i][1])
        Tp_large = float(vin_sync_5days[i][2])
        Dir_large = float(vin_sync_5days[i][3])

        # Utilisation exclusive d'OS2NS_uni_pluriel
        pred_low, pred_high = OS2NS_uni_pluriel(points, Hs_large, Tp_large, Dir_large, 0, False)

        S_H = pred_high[:, :2].astype(float).flatten()
        S_L = pred_low[:, :2].astype(float).flatten()

        # Lois physiques de la fonction de transfert (combinaison par le coef de marée)
        Hs_recon = coef * S_H[0] + (1 - coef) * S_L[0]
        Tp_recon = coef * S_H[1] + (1 - coef) * S_L[1]

        pred_Hs_list.append(Hs_recon)
        pred_Tp_list.append(Tp_recon)
        print(i/len(v2_5days))

    pred_Hs = np.array(pred_Hs_list)
    pred_Tp = np.array(pred_Tp_list)

    # Nettoyage des NaN pour assurer la stabilité des graphiques et calculs
    valid_mask = ~np.isnan(pred_Hs) & ~np.isnan(pred_Tp)
    dates_plot = real_dates[valid_mask]
    real_Hs_plot = real_Hs[valid_mask]
    real_Tp_plot = real_Tp[valid_mask]
    pred_Hs_plot = pred_Hs[valid_mask]
    pred_Tp_plot = pred_Tp[valid_mask]

    print(f"   → Points valides calculés : {len(dates_plot)}")

    # 4. Calcul et affichage des métriques de performance
    print("\nMÉTRIQUES DE PERFORMANCE (FONCTION DE TRANSFERT - 5 JOURS)")
    print("=" * 70)

    mse_hs = mean_squared_error(real_Hs_plot, pred_Hs_plot)
    mse_tp = mean_squared_error(real_Tp_plot, pred_Tp_plot)
    r2_hs = r2_score(real_Hs_plot, pred_Hs_plot)
    r2_tp = r2_score(real_Tp_plot, pred_Tp_plot)

    print(f"Hs → MSE : {mse_hs:.6f} m² | RMSE : {np.sqrt(mse_hs):.4f} m | R² : {r2_hs:.4f}")
    print(f"Tp → MSE : {mse_tp:.6f} s² | RMSE : {np.sqrt(mse_tp):.4f} s | R² : {r2_tp:.4f}")

    # 5. Génération des graphiques
    print("\n[5/5] Génération des graphiques comparatifs...")
    
    plot_comparison(
        dates_plot, real_Hs_plot, real_Tp_plot,
        pred_Hs_plot, pred_Tp_plot,
        title=f"Sonde 2 : Données Réelles vs Fonction de Transfert\n{first_date.strftime('%Y-%m-%d')} à {last_date.strftime('%Y-%m-%d')}",
        save_path="comparison_sonde2_5days_transfer.png"
    )

    plot_scatter_comparison(
        real_Hs_plot, real_Tp_plot,
        pred_Hs_plot, pred_Tp_plot,
        save_path="scatter_sonde2_5days_transfer.png"
    )

    print("\nEXECUTION TERMINÉE AVEC SUCCÈS !")
    print("=" * 70)


if False:
    main()