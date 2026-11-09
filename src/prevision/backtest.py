"""
Validation glissante (« backtest ») : chaque dimanche soir, chaque modèle prévoit les 7 jours suivants
en ne connaissant que le passé. On compare ensuite les prévisions à la consommation réelle.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .models import Forecaster

HORIZON = 7
REFIT_EVERY = 4  # les modèles lents sont réestimés toutes les 4 semaines, et simplement mis à jour entre-temps
COMBINED = "Combinaison (moyenne)"
COMBINED_MEMBERS = ["SARIMAX + Fourier", "Prophet", "LightGBM"]


def weekly_origins(y: pd.Series, start: str, end: str) -> pd.DatetimeIndex:
    """Tous les dimanches entre start et end, à condition que la semaine suivante soit connue."""
    sundays = pd.date_range(start, end, freq="W-SUN")
    return sundays[sundays + pd.Timedelta(days=HORIZON) <= y.index[-1]]


def run_backtest(y: pd.Series, models: list[Forecaster], origins: pd.DatetimeIndex, verbose: bool = True) -> pd.DataFrame:
    rows, timings = [], {m.name: 0.0 for m in models}
    for i, origin in enumerate(origins):
        train = y.loc[:origin]
        actual = y.loc[origin + pd.Timedelta(days=1): origin + pd.Timedelta(days=HORIZON)]
        for m in models:
            t0 = time.perf_counter()
            (m.fit if i % REFIT_EVERY == 0 else m.update)(train)
            fc = m.predict(HORIZON)
            timings[m.name] += time.perf_counter() - t0
            fc["reel"] = actual.to_numpy()
            fc["horizon"] = np.arange(1, HORIZON + 1)
            fc["origine"] = origin
            fc["modele"] = m.name
            fc["famille"] = m.family
            rows.append(fc.rename_axis("date").reset_index())
        if verbose and (i + 1) % 13 == 0:
            print(f"  {i + 1}/{len(origins)} semaines évaluées")
    out = pd.concat(rows, ignore_index=True)

    # combinaison : moyenne des prévisions de 3 modèles de familles différentes (choisis à l'avance)
    comb = (out[out["modele"].isin(COMBINED_MEMBERS)]
            .groupby(["date", "origine", "horizon"], as_index=False)
            .agg(prevision=("prevision", "mean"), borne_basse=("borne_basse", "mean"),
                 borne_haute=("borne_haute", "mean"), reel=("reel", "first")))
    comb["modele"], comb["famille"] = COMBINED, "Combinaison"
    out = pd.concat([out, comb], ignore_index=True)
    out.attrs["timings"] = timings
    return out


# ─────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────
def mase_scale(y: pd.Series, end: str) -> float:
    """Erreur moyenne du naïf saisonnier sur l'historique d'entraînement : l'unité du MASE."""
    hist = y.loc[:end]
    return float((hist - hist.shift(7)).abs().mean())


def metrics(df: pd.DataFrame, scale: float) -> dict:
    err = df["prevision"] - df["reel"]
    inside = (df["reel"] >= df["borne_basse"]) & (df["reel"] <= df["borne_haute"])
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((err.abs() / df["reel"]).mean() * 100),
        "mase": float(err.abs().mean() / scale),
        "biais": float(err.mean()),
        "couverture_80": float(inside.mean() * 100),
        "largeur_intervalle": float((df["borne_haute"] - df["borne_basse"]).mean()),
    }


def summary_table(bt: pd.DataFrame, scale: float, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["modele", "famille"] + (by or [])
    rows = [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **metrics(g, scale)}
            for k, g in bt.groupby(keys, sort=False)]
    return pd.DataFrame(rows)


def diebold_mariano(bt: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """
    Test de Diebold-Mariano : l'écart d'erreur entre deux modèles est-il réel ou dû au hasard ?
    On compare l'erreur absolue moyenne de chaque semaine (une valeur par semaine, donc peu corrélées),
    avec une correction de Newey-West (1 retard) pour l'autocorrélation restante.
    """
    from scipy import stats

    def weekly(name):
        d = bt[bt["modele"] == name]
        return (d["prevision"] - d["reel"]).abs().groupby(d["origine"]).mean()

    d = (weekly(model_a) - weekly(model_b)).dropna().to_numpy()
    n = len(d)
    dc = d - d.mean()
    gamma0 = (dc @ dc) / n
    gamma1 = (dc[1:] @ dc[:-1]) / n
    var = (gamma0 + 2 * 0.5 * gamma1) / n
    stat = d.mean() / np.sqrt(var)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return {"modele_a": model_a, "modele_b": model_b, "ecart_mae": float(d.mean()), "statistique": float(stat),
            "p_value": float(p), "semaines": int(n)}
