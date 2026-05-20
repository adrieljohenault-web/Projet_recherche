
"""
Script de visualisation des résultats du modèle IA2 pour la sonde 2.
Version optimisée : calcul uniquement sur les 5 premiers jours.
"""

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
import pickle

from scipy.optimize import nnls
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Imports des modules du projet
from variables_globales import *
from Fonction_de_transfert import *
from sonde_donnee_formatage import v1, v2, v3, vin, vin_sync


# =============================================================================
# ÉTAPE 0 : Chargement des coefficients a,b depuis le fichier CSV
# =============================================================================

def load_coefs_sonde2(filepath="coefs_sonde2.csv", start_date=None, end_date=None):
    """Charge les coefficients a,b pour la sonde 2 depuis le CSV, avec filtre de dates."""
    result = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M")
            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue
            result.append([
                dt,
                np.float64(row["coef_a"]),
                np.float64(row["coef_b"])
            ])
    return np.array(result, dtype=object)


# =============================================================================
# ÉTAPE 1 : Construction des features X
# =============================================================================

def get_mean_depth(v_sonde):
    """Calcule la profondeur moyenne d'une sonde."""
    depth = v_sonde[:, 3]
    return np.mean(depth)


def compute_marnage(v_large, target_dates, window_days=2):
    """Calcule le marnage et l'interpole aux dates cibles."""
    source_dates = v_large[:, 0]
    dpts = v_large[:, 4].astype(float)

    marnage_source = []
    for i, date in enumerate(source_dates):
        cutoff = date - datetime.timedelta(days=window_days)
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

    return np.column_stack((target_dates, marnage_interp))


def get_X_for_dates(vin_sync, marn_data, depth):
    # 1. Extraction des dates
    dates_vin = vin_sync[:, 0]
    dates_marn = marn_data[:, 0]
    
    # 2. Intersection des dates communes
    dates_communes = sorted(set(dates_vin) & set(dates_marn))
    
    # 3. Filtrage INDÉPENDANT des deux arrays
    mask_vin = np.array([d in dates_communes for d in dates_vin])
    mask_marn = np.array([d in dates_communes for d in dates_marn])
    
    vin_sync_clean = vin_sync[mask_vin]
    marn_val_clean = marn_data[mask_marn, 1]
    
    # 4. Vérification cohérence
    assert len(vin_sync_clean) == len(marn_val_clean)
    
    # 5. Assemblage
    common = np.column_stack((vin_sync_clean, marn_val_clean))
    X = np.column_stack((common, np.full(len(common), depth)))
    return X


# =============================================================================
# ÉTAPE 2 : Entraînement du modèle XGBoost (ou chargement)
# =============================================================================

def train_or_load_model(X_train, y_train, model_path="model_sonde2_5days.pkl"):
    """Entraîne le modèle XGBoost multi-sortie ou charge un modèle existant."""
    if os.path.exists(model_path):
        print(f"Chargement du modèle existant : {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    print("Entraînement du modèle XGBoost...")

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=2
    )

    multi_model = MultiOutputRegressor(xgb_model)
    multi_model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(multi_model, f)
    print(f"Modèle sauvegardé : {model_path}")

    return multi_model


# =============================================================================
# ÉTAPE 3 : Visualisation
# =============================================================================
def plot_comparison(dates, real_Hs, real_Tp, pred_Hs, pred_Tp,
                    title="Sonde 2 : Comparaison Données Réelles vs Modèle (5 jours)",
                    save_path="comparison_sonde2_5days.png"):
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

    # Modèle IA : ligne brisée
    ax1.plot(
        dates,
        pred_Hs,
        'r--',
        label='Hs Modèle (prédit)',
        alpha=0.8,
        linewidth=1.5
    )

    # Modèle IA : points
    ax1.scatter(
        dates,
        pred_Hs,
        color='red',
        alpha=0.7,
        s=15
    )

    ax1.set_ylabel('Hs [m]', fontsize=12)
    ax1.set_title(
        f'{title} - Hauteur Significative',
        fontsize=14,
        fontweight='bold'
    )
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

    # Modèle IA : ligne brisée
    ax2.plot(
        dates,
        pred_Tp,
        'r--',
        label='Tp Modèle (prédit)',
        alpha=0.8,
        linewidth=1.5
    )

    # Modèle IA : points
    ax2.scatter(
        dates,
        pred_Tp,
        color='red',
        alpha=0.7,
        s=15
    )

    ax2.set_ylabel('Tp [s]', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(
        f'{title} - Période de Pic',
        fontsize=14,
        fontweight='bold'
    )
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Graphique sauvegardé : {save_path}")

    plt.show()

def plot_scatter_comparison(real_Hs, real_Tp, pred_Hs, pred_Tp,
                            save_path="scatter_sonde2_5days.png"):
    """Crée des scatter plots réel vs prédit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter Hs
    ax1 = axes[0]
    ax1.scatter(real_Hs, pred_Hs, alpha=0.5, c='blue', s=20)
    max_hs = max(max(real_Hs), max(pred_Hs))
    ax1.plot([0, max_hs], [0, max_hs], 'r--', lw=2, label='y=x (parfait)')
    ax1.set_xlabel('Hs Réel [m]', fontsize=12)
    ax1.set_ylabel('Hs Prédit [m]', fontsize=12)
    ax1.set_title('Hs : Réel vs Prédit', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter Tp
    ax2 = axes[1]
    ax2.scatter(real_Tp, pred_Tp, alpha=0.5, c='green', s=20)
    max_tp = max(max(real_Tp), max(pred_Tp))
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
# MAIN : Exécution sur 5 premiers jours uniquement
# =============================================================================

def main():
    print("=" * 70)
    print("VISUALISATION MODÈLE IA2 - SONDE 2 (5 PREMIERS JOURS)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 0. Détermination de la fenêtre temporelle : 5 premiers jours
    # -------------------------------------------------------------------------
    print("[0/6] Détermination de la fenêtre des 5 premiers jours...")

    # On prend la première date disponible dans v2 (sonde 2)
    first_date = min(v2[:, 0])
    last_date = first_date + datetime.timedelta(days=5)

    print(f"   → Première date sonde 2 : {first_date}")
    print(f"   → Fenêtre 5 jours       : {first_date} à {last_date}")

    # -------------------------------------------------------------------------
    # 1. Chargement des coefficients a,b (filtrés sur 5 jours)
    # -------------------------------------------------------------------------
    print("[1/6] Chargement des coefficients a,b (5 jours)...")
    coefs_sonde2 = load_coefs_sonde2("coefs_sonde2.csv", start_date=first_date, end_date=last_date)
    print(f"   → {len(coefs_sonde2)} coefficients chargés")
    if len(coefs_sonde2) > 0:
        print(f"   → Période : {coefs_sonde2[0,0]} à {coefs_sonde2[-1,0]}")

    # -------------------------------------------------------------------------
    # 2. Récupération des données réelles de la sonde 2 (filtrées 5 jours)
    # -------------------------------------------------------------------------
    print("[2/6] Récupération des données réelles de la sonde 2 (5 jours)...")

    # Filtrage v2 sur les 5 premiers jours
    mask_v2_5days = np.array([
        first_date <= d <= last_date for d in v2[:, 0]
    ])
    v2_5days = v2[mask_v2_5days]

    # Intersection des dates entre coefs et sonde
    dates_coefs = set(coefs_sonde2[:, 0])
    dates_sonde = set(v2_5days[:, 0])
    dates_communes = sorted(dates_coefs & dates_sonde)
    print(f"   → Dates communes : {len(dates_communes)}")

    # Filtrage final
    mask_v2 = np.array([d in dates_communes for d in v2_5days[:, 0]])
    v2_filtered = v2_5days[mask_v2]

    mask_coefs = np.array([d in dates_communes for d in coefs_sonde2[:, 0]])
    coefs_filtered = coefs_sonde2[mask_coefs]

    real_dates = v2_filtered[:, 0]
    real_Hs = v2_filtered[:, 1].astype(float)
    real_Tp = v2_filtered[:, 2].astype(float)

    # -------------------------------------------------------------------------
    # 3. Construction des features X pour le modèle
    # -------------------------------------------------------------------------
    print("[3/6] Construction des features X...")

    depth_v2 = get_mean_depth(v2)
    marnage = compute_marnage(vin, real_dates)

    # Construction de X complet [date, Hs, Tp, Dir, Marnage, Depth]
    X_full = get_X_for_dates(vin_sync, marnage, depth_v2)

    # Vérification synchronisation
    X_dates = X_full[:, 0]
    y_dates = coefs_filtered[:, 0]

    # On ne garde que les dates communes entre X et y
    common_dates_mask_X = np.array([d in dates_communes for d in X_dates])
    common_dates_mask_y = np.array([d in dates_communes for d in y_dates])

    X_full_sync = X_full[common_dates_mask_X]
    coefs_sync = coefs_filtered[common_dates_mask_y]

    # Features sans date pour le modèle
    X_features = X_full_sync[:, 1:].astype(float)

    # Cibles y (coefficients a,b)
    y_targets = coefs_sync[:, 1:].astype(float)

    print(f"   → Features X shape : {X_features.shape}")
    print(f"   → Cibles y shape   : {y_targets.shape}")

    # -------------------------------------------------------------------------
    # 4. Entraînement / Chargement du modèle
    # -------------------------------------------------------------------------
    print("[4/6] Entraînement/Chargement du modèle XGBoost...")

    # Split train/test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_features, y_targets, np.arange(len(X_features)),
        test_size=0.2, random_state=42
    )

    model = train_or_load_model(X_train, y_train, "model_sonde2_5days.pkl")

    # Évaluation sur le test set
    y_pred_test = model.predict(X_test)
    print("Évaluation sur le set de test :")
    for i, name in enumerate(['coef_a', 'coef_b']):
        mse = mean_squared_error(y_test[:, i], y_pred_test[:, i])
        r2 = r2_score(y_test[:, i], y_pred_test[:, i])
        print(f"   → {name} : MSE={mse:.4f}, R²={r2:.4f}")

    # -------------------------------------------------------------------------
    # 5. Prédiction des coefficients a,b pour TOUTES les dates (5 jours)
    # -------------------------------------------------------------------------
    print("[5/6] Prédiction des coefficients a,b pour les dates (5 jours)...")

    y_pred_all = model.predict(X_features)

    # -------------------------------------------------------------------------
    # 6. Reconstruction de Hs et Tp à partir des coefficients prédits
    # -------------------------------------------------------------------------
    print("[6/6] Reconstruction de Hs et Tp depuis les coefficients prédits...")

    pred_Hs_list = []
    pred_Tp_list = []

    for i, date in enumerate(real_dates):
        a_pred = y_pred_all[i, 0]
        b_pred = y_pred_all[i, 1]

        # Récupération conditions au large
        mask = vin_sync[:, 0] == date
        if not np.any(mask):
            pred_Hs_list.append(np.nan)
            pred_Tp_list.append(np.nan)
            continue

        idx = np.where(mask)[0][0]
        Hs_large = float(vin_sync[idx, 1])
        Tp_large = float(vin_sync[idx, 2])
        Dir_large = float(vin_sync[idx, 3])

        # Fonction de transfert
        points = points_and_weights[1]
        pred_low, pred_high = OS2NS_uni_pluriel(points, Hs_large, Tp_large, Dir_large, 0, False)

        S_H = pred_high[:, :2].astype(float).flatten()
        S_L = pred_low[:, :2].astype(float).flatten()

        Hs_recon = a_pred * S_H[0] + b_pred * S_L[0]
        Tp_recon = a_pred * S_H[1] + b_pred * S_L[1]

        pred_Hs_list.append(Hs_recon)
        pred_Tp_list.append(Tp_recon)
        print(i/len(real_dates))

    pred_Hs = np.array(pred_Hs_list)
    pred_Tp = np.array(pred_Tp_list)

    # Suppression des NaN pour les graphiques
    valid_mask = ~np.isnan(pred_Hs) & ~np.isnan(pred_Tp)
    dates_plot = real_dates[valid_mask]
    real_Hs_plot = real_Hs[valid_mask]
    real_Tp_plot = real_Tp[valid_mask]
    pred_Hs_plot = pred_Hs[valid_mask]
    pred_Tp_plot = pred_Tp[valid_mask]

    print(f"   → Points valides pour le graphique : {len(dates_plot)}")

    # -------------------------------------------------------------------------
    # 7. Calcul des métriques de performance
    # -------------------------------------------------------------------------
    print("MÉTRIQUES DE PERFORMANCE (5 JOURS)")
    print("=" * 70)

    mse_hs = mean_squared_error(real_Hs_plot, pred_Hs_plot)
    mse_tp = mean_squared_error(real_Tp_plot, pred_Tp_plot)
    r2_hs = r2_score(real_Hs_plot, pred_Hs_plot)
    r2_tp = r2_score(real_Tp_plot, pred_Tp_plot)

    print(f"Hs → MSE : {mse_hs:.6f} m² | RMSE : {np.sqrt(mse_hs):.4f} m | R² : {r2_hs:.4f}")
    print(f"Tp → MSE : {mse_tp:.6f} s² | RMSE : {np.sqrt(mse_tp):.4f} s | R² : {r2_tp:.4f}")

    # -------------------------------------------------------------------------
    # 8. Visualisation
    # -------------------------------------------------------------------------

    print("GÉNÉRATION DES GRAPHIQUES")
    print("=" * 70)

    # Graphique temporel
    plot_comparison(
        dates_plot, real_Hs_plot, real_Tp_plot,
        pred_Hs_plot, pred_Tp_plot,
        title=f"Sonde 2 : Données Réelles vs Modèle IA2\n{first_date.strftime('%Y-%m-%d')} à {last_date.strftime('%Y-%m-%d')}",
        save_path="comparison_sonde2_5days.png"
    )

    # Scatter plots
    plot_scatter_comparison(
        real_Hs_plot, real_Tp_plot,
        pred_Hs_plot, pred_Tp_plot,
        save_path="scatter_sonde2_5days.png"
    )

    print("TERMINÉ !")
    print("=" * 70)
    print("Fichiers générés :")
    print("  - comparison_sonde2_5days.png : Comparaison temporelle (5 jours)")
    print("  - scatter_sonde2_5days.png    : Scatter plots réel vs prédit (5 jours)")
    print("  - model_sonde2_5days.pkl      : Modèle entraîné (sauvegarde)")


if True:
    main()