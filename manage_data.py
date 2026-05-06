"""
consolidate_data.py
-------------------
Consolide tous les fichiers D3D_res****_SH.txt / _SL.txt en un seul
fichier binaire NumPy (.npz) et, optionnellement, en CSV.

Pourquoi .npz et pas seulement CSV ?
  - Lecture ~10-50x plus rapide (format binaire, pas de parsing texte)
  - Chargement direct en np.ndarray, sans pandas ni re-parsing
  - Fichiers plus légers (float32 au lieu de texte)

Structure des tableaux produits :
  SH : shape (n_valeurs_calc, n_sortie, 8)   # marée haute
  SL : shape (n_valeurs_calc, n_sortie, 8)   # marée basse

  Colonnes (axe 2) : Hs, Tp, Tm01, Dp, Dm, DSpr, WD, Qb

Usage :
  python consolidate_data.py            # produit seulement le .npz
  python consolidate_data.py --csv      # produit aussi le CSV (plus lent)
"""

import os
import sys
import time
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# À ADAPTER selon ton environnement (ou importer depuis variables_globales.py)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from variables_globales import path, n_valeurs_calc, n_sortie
except ImportError:
    # Valeurs de repli pour test autonome
    path = "."
    n_valeurs_calc = 100   # nombre de simulations
    n_sortie = 36226       # nombre de points de maillage (lignes hors header)

COLS = ["Hs_m", "Tp_s", "Tm01_s", "Dp_degN", "Dm_degN", "DSpr_deg", "WD_m", "Qb"]
N_COLS = len(COLS)  # 8

DIR_SH = os.path.join(path, "Delft3D_sorties_gamma04", "SH")
DIR_SL = os.path.join(path, "Delft3D_sorties_gamma04", "SL")
OUT_NPZ = os.path.join(path, "training_data.npz")
OUT_CSV_SH = os.path.join(path, "training_data_SH.csv")
OUT_CSV_SL = os.path.join(path, "training_data_SL.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def fmt_index(n: int) -> str:
    return str(n).zfill(4)


def read_one_file(filepath: str) -> np.ndarray:
    """Lit un fichier SH ou SL et retourne un tableau (n_sortie, 8) float32."""
    with open(filepath, "r") as f:
        lines = f.readlines()[1:]           # on saute le header
    data = np.array(
        [list(map(float, line.split())) for line in lines],
        dtype=np.float32,
    )
    return data


def progress(i: int, total: int, t0: float):
    elapsed = time.time() - t0
    pct = i / total
    eta = (elapsed / pct * (1 - pct)) if pct > 0 else 0
    bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
    print(f"\r  [{bar}] {i}/{total}  ETA {eta:.0f}s ", end="", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Consolidation principale
# ──────────────────────────────────────────────────────────────────────────────

def build_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Lit tous les fichiers et retourne (SH, SL) de shape (n_valeurs_calc, n_sortie, 8).
    """
    print(f"Lecture de {n_valeurs_calc} simulations × 2 fichiers ({n_sortie} points chacun)…")
    SH = np.empty((n_valeurs_calc, n_sortie, N_COLS), dtype=np.float32)
    SL = np.empty((n_valeurs_calc, n_sortie, N_COLS), dtype=np.float32)

    t0 = time.time()
    for idx in range(n_valeurs_calc):
        i = idx + 1                         # les fichiers commencent à 0001
        tag = fmt_index(i)

        path_sh = os.path.join(DIR_SH, f"D3D_res{tag}_SH.txt")
        path_sl = os.path.join(DIR_SL, f"D3D_res{tag}_SL.txt")

        SH[idx] = read_one_file(path_sh)
        SL[idx] = read_one_file(path_sl)

        if (idx + 1) % max(1, n_valeurs_calc // 100) == 0 or idx == n_valeurs_calc - 1:
            progress(idx + 1, n_valeurs_calc, t0)

    print(f"\n  ✓ Lecture terminée en {time.time()-t0:.1f}s")
    return SH, SL


def save_npz(SH: np.ndarray, SL: np.ndarray):
    """Sauvegarde en binaire NumPy compressé."""
    print(f"Sauvegarde .npz → {OUT_NPZ}")
    t0 = time.time()
    np.savez_compressed(OUT_NPZ, SH=SH, SL=SL)
    size_mb = os.path.getsize(OUT_NPZ) / 1e6
    print(f"  ✓ {size_mb:.1f} Mo en {time.time()-t0:.1f}s")


def save_csv(SH: np.ndarray, SL: np.ndarray):
    """
    Sauvegarde en CSV.
    Format : sim_index (1-based), point_index (0-based), puis les 8 colonnes.
    → Une ligne = un point de maillage pour une simulation donnée.
    """
    header = "sim_index,point_index," + ",".join(COLS)

    for name, arr, out_path in [("SH", SH, OUT_CSV_SH), ("SL", SL, OUT_CSV_SL)]:
        print(f"Sauvegarde CSV {name} → {out_path}  (peut être long…)")
        t0 = time.time()
        with open(out_path, "w") as f:
            f.write(header + "\n")
            for sim_idx in range(n_valeurs_calc):
                for pt_idx in range(n_sortie):
                    row = f"{sim_idx+1},{pt_idx}," + ",".join(
                        f"{v:.6g}" for v in arr[sim_idx, pt_idx]
                    )
                    f.write(row + "\n")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  ✓ {size_mb:.1f} Mo en {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de chargement (à copier dans IA.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_from_npz(npz_path: str = OUT_NPZ) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les données consolidées.
    Retourne (SH, SL) de shape (n_valeurs_calc, n_sortie, 8).

    Exemple d'usage dans IA.py :
        from consolidate_data import load_from_npz
        SH, SL = load_from_npz()
        y_trainh = SH   # (n_valeurs_calc, n_sortie, 8)
        y_trainl = SL
    """
    data = np.load(npz_path)
    return data["SH"], data["SL"]


def load_from_csv(csv_sh: str = OUT_CSV_SH, csv_sl: str = OUT_CSV_SL):
    """Charge depuis les CSV (plus lent, usage déconseillé en production)."""
    import pandas as pd
    df_sh = pd.read_csv(csv_sh)
    df_sl = pd.read_csv(csv_sl)

    SH = df_sh[COLS].values.reshape(n_valeurs_calc, n_sortie, N_COLS).astype(np.float32)
    SL = df_sl[COLS].values.reshape(n_valeurs_calc, n_sortie, N_COLS).astype(np.float32)
    return SH, SL


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    also_csv = "--csv" in sys.argv

    SH, SL = build_arrays()
    save_npz(SH, SL)

    if also_csv:
        save_csv(SH, SL)
    else:
        print("(Passe --csv pour générer aussi les fichiers CSV)")

    # ── Vérification rapide ──────────────────────────────────────────────────
    print("\nVérification : rechargement du .npz…")
    t0 = time.time()
    SH2, SL2 = load_from_npz()
    print(f"  ✓ Rechargé en {time.time()-t0:.3f}s  |  SH {SH2.shape}  SL {SL2.shape}")
    assert np.allclose(SH, SH2, atol=1e-4), "Erreur de cohérence SH !"
    assert np.allclose(SL, SL2, atol=1e-4), "Erreur de cohérence SL !"
    print("  ✓ Données cohérentes.")