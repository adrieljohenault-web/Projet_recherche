"""
Script d'optimisation des hyperparamètres XGBoost pour le projet IA2
Ce script teste n_estimators, learning_rate et max_depth
et génère des graphiques de MSE et R² pour chaque paramètre.

Intégration : Copie ce fichier dans ton dossier de travail et exécute :
    python hyperparam_optimization.py
"""

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION - Modifie ces chemins selon ton projet
# ============================================================================

# Si tu utilises directement les variables de ton projet IA2.py :
# from IA2 import get_X, get_y  # Décommente si tu importes depuis IA2.py

# Sinon, charge depuis les CSV générés par IA2.py
def load_data_from_csv():
    """
    Charge les données depuis les fichiers CSV de coefficients.
    Adapte les chemins selon ton projet.
    """
    # Exemple de chargement - adapte selon ta structure
    # X = np.loadtxt("X_data.csv", delimiter=",")
    # y = np.loadtxt("y_data.csv", delimiter=",")

    # Pour l'instant, placeholder - remplace par ton chargement réel
    raise NotImplementedError(
        "Modifie cette fonction pour charger tes vraies données.\n"
        "Soit importe X et y depuis IA2.py, soit charge tes CSV."
    )

# ============================================================================
# PARAMÈTRES À TESTER
# ============================================================================

PARAM_GRID = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 6, 8, 10]
}

DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': 2
}

# ============================================================================
# FONCTIONS PRINCIPALES
# ============================================================================

def evaluate_model(X_train, X_test, y_train, y_test, params):
    """
    Entraîne un MultiOutputRegressor XGBoost et retourne MSE/R² pour Hs et Tp.
    """
    xgb_model = xgb.XGBRegressor(**params)
    multi_model = MultiOutputRegressor(xgb_model)
    multi_model.fit(X_train, y_train)
    y_pred = multi_model.predict(X_test)

    mse_h = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_t = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_h = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_t = r2_score(y_test[:, 1], y_pred[:, 1])

    return mse_h, mse_t, r2_h, r2_t


def optimize_hyperparams(X, y, test_size=0.2, output_dir="./figures"):
    """
    Teste chaque hyperparamètre individuellement et génère les graphiques.

    Args:
        X: Features (sans dates)
        y: Targets [coef_a, coef_b] (sans dates)
        test_size: Fraction pour le test set
        output_dir: Dossier de sauvegarde des figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Données : Train={X_train.shape}, Test={X_test.shape}")
    print(f"Targets : Hs (coef_a) et Tp (coef_b)")
    print("=" * 60)

    results = {}
    best_params = {}

    for param_name, param_values in PARAM_GRID.items():
        print(f"\n📊 Test du paramètre : {param_name}")
        print("-" * 40)

        mse_h_list, mse_t_list = [], []
        r2_h_list, r2_t_list = [], []

        for val in param_values:
            current_params = DEFAULT_PARAMS.copy()
            current_params[param_name] = val

            mse_h, mse_t, r2_h, r2_t = evaluate_model(
                X_train, X_test, y_train, y_test, current_params
            )

            mse_h_list.append(mse_h)
            mse_t_list.append(mse_t)
            r2_h_list.append(r2_h)
            r2_t_list.append(r2_t)

            print(f"  {param_name:15s} = {str(val):6s} | "
                  f"MSE_Hs={mse_h:.5f}  MSE_Tp={mse_t:.5f} | "
                  f"R²_Hs={r2_h:.4f}  R²_Tp={r2_t:.4f}")

        results[param_name] = {
            'values': param_values,
            'mse_h': mse_h_list,
            'mse_t': mse_t_list,
            'r2_h': r2_h_list,
            'r2_t': r2_t_list
        }

        # Meilleures valeurs pour ce paramètre
        best_h = param_values[np.argmin(mse_h_list)]
        best_t = param_values[np.argmin(mse_t_list)]
        best_params[param_name] = {'Hs': best_h, 'Tp': best_t}
        print(f"  → Meilleur pour Hs : {best_h} | Meilleur pour Tp : {best_t}")

    # ============================================================================
    # GÉNÉRATION DES GRAPHIQUES
    # ============================================================================

    print("\n" + "=" * 60)
    print("🎨 Génération des graphiques...")

    for param_name in PARAM_GRID.keys():
        data = results[param_name]
        values = data['values']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Optimisation du paramètre : {param_name}', 
                     fontsize=16, fontweight='bold')

        # --- Graphique MSE ---
        ax1 = axes[0]
        ax1.plot(values, data['mse_h'], 'o-', color='#e74c3c', 
                linewidth=2.5, markersize=9, label='MSE Hs (coef_a)')
        ax1.plot(values, data['mse_t'], 's-', color='#3498db', 
                linewidth=2.5, markersize=9, label='MSE Tp (coef_b)')
        ax1.set_xlabel(param_name, fontsize=13)
        ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
        ax1.set_title('Évolution de la MSE', fontsize=14)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(labelsize=11)

        # Annotation des minima
        best_h_idx = np.argmin(data['mse_h'])
        best_t_idx = np.argmin(data['mse_t'])
        ax1.scatter(values[best_h_idx], data['mse_h'][best_h_idx], 
                   s=150, c='#e74c3c', marker='*', zorder=5, edgecolors='black')
        ax1.scatter(values[best_t_idx], data['mse_t'][best_t_idx], 
                   s=150, c='#3498db', marker='*', zorder=5, edgecolors='black')

        # --- Graphique R² ---
        ax2 = axes[1]
        ax2.plot(values, data['r2_h'], 'o-', color='#e74c3c', 
                linewidth=2.5, markersize=9, label='R² Hs (coef_a)')
        ax2.plot(values, data['r2_t'], 's-', color='#3498db', 
                linewidth=2.5, markersize=9, label='R² Tp (coef_b)')
        ax2.set_xlabel(param_name, fontsize=13)
        ax2.set_ylabel('Coefficient de détermination R²', fontsize=13)
        ax2.set_title('Évolution du R²', fontsize=14)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(labelsize=11)

        # Annotation des maxima
        best_r2_h_idx = np.argmax(data['r2_h'])
        best_r2_t_idx = np.argmax(data['r2_t'])
        ax2.scatter(values[best_r2_h_idx], data['r2_h'][best_r2_h_idx], 
                   s=150, c='#e74c3c', marker='*', zorder=5, edgecolors='black')
        ax2.scatter(values[best_r2_t_idx], data['r2_t'][best_r2_t_idx], 
                   s=150, c='#3498db', marker='*', zorder=5, edgecolors='black')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filepath = os.path.join(output_dir, f'xgboost_{param_name}.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"  ✅ Sauvegardé : {filepath}")
        plt.close()

    # ============================================================================
    # TABLEAU RÉCAPITULATIF
    # ============================================================================

    print("\n" + "=" * 60)
    print("📋 TABLEAU RÉCAPITULATIF DES MEILLEURS PARAMÈTRES")
    print("=" * 60)
    print(f"{'Paramètre':<15} {'Meilleur Hs':<15} {'Meilleur Tp':<15}")
    print("-" * 45)
    for param, vals in best_params.items():
        print(f"{param:<15} {str(vals['Hs']):<15} {str(vals['Tp']):<15}")

    print("\n💡 Conseil : Si Hs et Tp préfèrent des valeurs différentes,")
    print("   teste une grille combinée (GridSearch) autour de ces valeurs.")

    return results, best_params


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if False:
    # ------------------------------------------------------------------------
    # OPTION 1 : Charger depuis ton projet IA2.py (RECOMMANDÉ)
    # ------------------------------------------------------------------------
    # Décommente et adapte selon ton projet :

    # from IA2 import get_X_y_wo_date, get_X, get_y  # ou importe tes fonctions
    # X_dates = get_X(...)  # ta fonction de construction de X
    # y_dates = get_y()     # ta fonction de chargement de y
    # X, y = get_X_y_wo_date(X_dates, y_dates)

    # ------------------------------------------------------------------------
    # OPTION 2 : Charger depuis des fichiers CSV
    # ------------------------------------------------------------------------
    # X = np.loadtxt("X_data.csv", delimiter=",")
    # y = np.loadtxt("y_data.csv", delimiter=",")

    # ------------------------------------------------------------------------
    # OPTION 3 : Données de démonstration (à supprimer)
    # ------------------------------------------------------------------------
    print("⚠️  Mode démonstration avec données synthétiques")
    print("   → Remplace cette section par le chargement de tes vraies données\n")

    np.random.seed(42)
    n_samples = 2000
    n_features = 5
    X = np.random.rand(n_samples, n_features)
    y = np.column_stack([
        0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 0.1, n_samples),
        0.4 * X[:, 2] + 0.2 * X[:, 3] + np.random.normal(0, 0.15, n_samples)
    ])

    # Lancer l'optimisation
    results, best_params = optimize_hyperparams(X, y, output_dir="./figures_xgboost")


if True:
    from IA2 import get_X_y_wo_date, X_dates, y_dates
    X, y = get_X_y_wo_date(X_dates, y_dates)
    results, best = optimize_hyperparams(X, y, output_dir="./figures")