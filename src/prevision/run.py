"""
Étude comparative complète :
  1. validation glissante de 7 modèles sur 104 semaines (2022-2023) ;
  2. métriques globales, par année, par saison, par horizon ; tests de Diebold-Mariano ;
  3. figures dans reports/figures/ ;
  4. prévision de la première semaine de 2024 avec le modèle retenu.

Lancer depuis src/ :  python -m prevision.run          (≈ 5 minutes)
                      python -m prevision.run --rapide (26 semaines, pour tester)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import (COMBINED, COMBINED_MEMBERS, diebold_mariano, mase_scale, run_backtest,
                       summary_table, weekly_origins)
from .data import ROOT, load_daily
from .models import all_models

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TEST_START, TEST_END = "2022-01-01", "2023-12-31"

COLORS = {
    "Naïf saisonnier": "#9aa0a6", "Naïf annuel ajusté": "#c4c7c5", "ETS (Holt-Winters)": "#8e6bbf",
    "SARIMAX + Fourier": "#1a73e8", "Prophet": "#e8710a", "LightGBM": "#188038", COMBINED: "#d93025",
}
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────
def fig_serie(y):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index, y, lw=0.4, color="#9aa0a6", label="Consommation journalière")
    ax.plot(y.index, y.rolling(365, center=True).mean(), lw=2.2, color="#d93025", label="Moyenne sur un an")
    ax.axvspan(pd.Timestamp("2020-03-17"), pd.Timestamp("2020-05-11"), color="#fbbc04", alpha=0.25)
    ax.text(pd.Timestamp("2020-04-14"), 2150, "Confinement", ha="center", fontsize=9)
    ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), color="#1a73e8", alpha=0.07)
    ax.text(pd.Timestamp("2023-01-01"), 2150, "Période de test\n(crise énergétique, sobriété)", ha="center", fontsize=9)
    ax.set_ylabel("GWh par jour")
    ax.set_title("Consommation d'électricité en France, 2012-2023 (source : RTE)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    _save(fig, "serie")


def fig_saisonnalites(y):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    months = ["janv", "févr", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc"]
    m = y.groupby(y.index.month).mean()
    axes[0].bar(months, m, color="#1a73e8")
    axes[0].set_title("Saisonnalité annuelle : le chauffage")
    axes[0].set_ylabel("GWh par jour (moyenne)")
    days = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
    d = y.groupby(y.index.dayofweek).mean()
    axes[1].bar(days, d, color=["#1a73e8"] * 5 + ["#9aa0a6"] * 2)
    axes[1].set_ylim(d.min() * 0.85, d.max() * 1.03)
    axes[1].set_title("Saisonnalité hebdomadaire : l'activité économique")
    _save(fig, "saisonnalites")


def fig_comparaison(table):
    t = table.sort_values("mape", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(t["modele"], t["mape"], color=[COLORS[m] for m in t["modele"]])
    for b, (_, r) in zip(bars, t.iterrows()):
        ax.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2,
                f"{r['mape']:.2f} %   (MAE {r['mae']:.0f} GWh)", va="center", fontsize=9)
    ax.set_xlim(0, t["mape"].max() * 1.45)
    ax.set_xlabel("Erreur moyenne en % (MAPE) — plus c'est court, mieux c'est")
    ax.set_title("Prévision à 7 jours : 104 semaines de test (2022-2023)")
    ax.grid(axis="y", visible=False)
    _save(fig, "comparaison")


def fig_horizon(by_h):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in [*COMBINED_MEMBERS, "ETS (Holt-Winters)", "Naïf saisonnier", COMBINED]:
        d = by_h[by_h["modele"] == m].sort_values("horizon")
        ax.plot(d["horizon"], d["mape"], marker="o", lw=2.6 if m == COMBINED else 1.6, color=COLORS[m], label=m)
    ax.set_xlabel("Horizon (jours après la prévision)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("L'erreur augmente avec l'horizon")
    ax.legend(frameon=False, fontsize=8.5)
    _save(fig, "erreur_horizon")


def fig_couverture(table):
    t = table.sort_values("couverture_80")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(t["modele"], t["couverture_80"], color=[COLORS[m] for m in t["modele"]])
    ax.axvline(80, color="black", ls="--", lw=1)
    ax.text(80.5, len(t) - 0.45, "cible : 80 %", fontsize=9)
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(r["couverture_80"] + 0.6, i, f"{r['couverture_80']:.0f} %  · largeur {r['largeur_intervalle']:.0f} GWh",
                va="center", fontsize=8.5)
    ax.set_xlim(40, 112)
    ax.set_xlabel("Part des jours où la vraie valeur tombe dans l'intervalle à 80 %")
    ax.set_title("Les intervalles de confiance sont-ils honnêtes ?")
    ax.grid(axis="y", visible=False)
    _save(fig, "couverture")


JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]


def fig_semaine(bt, origin, name, title):
    d = bt[bt["origine"] == origin]
    real = d[d["modele"] == COMBINED].set_index("date")["reel"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    c = d[d["modele"] == COMBINED].set_index("date")
    ax.fill_between(c.index, c["borne_basse"], c["borne_haute"], color=COLORS[COMBINED], alpha=0.12,
                    label="Intervalle à 80 % (combinaison)")
    for m in ["Naïf saisonnier", *COMBINED_MEMBERS, COMBINED]:
        s = d[d["modele"] == m].set_index("date")["prevision"]
        ax.plot(s.index, s, marker=".", lw=2.4 if m == COMBINED else 1.3, color=COLORS[m], label=m,
                ls=":" if m == "Naïf saisonnier" else "-")
    ax.plot(real.index, real, color="black", lw=2.2, marker="o", label="Consommation réelle")
    ax.set_xticks(real.index)
    ax.set_xticklabels([f"{JOURS[x.dayofweek]} {x:%d/%m}" for x in real.index])
    ax.set_ylabel("GWh par jour")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, name)


def weekly_errors(bt, model=COMBINED):
    d = bt[bt["modele"] == model]
    return (d["prevision"] - d["reel"]).abs().groupby(d["origine"]).mean().sort_values()


def fig_prevision(y, fc):
    hist = y.iloc[-28:]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(hist.index, hist, color="black", lw=1.8, label="Consommation réelle")
    ax.fill_between(fc.index, fc["borne_basse"], fc["borne_haute"], color=COLORS[COMBINED], alpha=0.15,
                    label="Intervalle à 80 %")
    ax.plot(fc.index, fc["prevision"], color=COLORS[COMBINED], lw=2.4, marker="o", label="Prévision")
    ax.set_ylabel("GWh par jour")
    ax.set_title("Prévision de la première semaine de 2024 (combinaison)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "prevision_2024")


# ─────────────────────────────────────────────────────────────
def main(fast: bool = False):
    FIG.mkdir(parents=True, exist_ok=True)
    y = load_daily()
    print(f"Données : {len(y)} jours, du {y.index[0]:%d/%m/%Y} au {y.index[-1]:%d/%m/%Y}")
    fig_serie(y)
    fig_saisonnalites(y)

    origins = weekly_origins(y, TEST_START, TEST_END)
    if fast:
        origins = origins[::4]
    print(f"Validation glissante : {len(origins)} semaines, {len(all_models())} modèles + 1 combinaison")
    t0 = time.time()
    bt = run_backtest(y, all_models(), origins)
    timings = bt.attrs["timings"]
    print(f"  terminé en {time.time() - t0:.0f} s")
    bt["annee"] = bt["date"].dt.year
    bt["saison"] = np.where(bt["date"].dt.month.isin([11, 12, 1, 2, 3]), "hiver", "été")

    scale = mase_scale(y, "2021-12-31")
    table = summary_table(bt, scale).sort_values("mae").reset_index(drop=True)
    by_year = summary_table(bt, scale, ["annee"])
    by_season = summary_table(bt, scale, ["saison"])
    by_h = summary_table(bt, scale, ["horizon"])

    best = table.iloc[0]["modele"]
    tests = [diebold_mariano(bt, best, m) for m in table["modele"] if m != best]

    # modèle retenu : le meilleur, et une vraie prévision au-delà des données
    print(f"Modèle retenu : {best}. Prévision de la semaine suivante...")
    members = [m for m in all_models() if m.name in (COMBINED_MEMBERS if best == COMBINED else [best])]
    fc = sum(m.fit(y).predict(7) for m in members) / len(members)
    fc.index.name = "date"
    fc.round(1).to_csv(REPORTS / "prevision_semaine.csv")

    fig_comparaison(table)
    fig_horizon(by_h)
    fig_couverture(table)
    werr = weekly_errors(bt)
    typical, worst = werr.index[len(werr) // 2], werr.index[-1]
    fig_semaine(bt, typical, "exemple_semaine",
                f"Une semaine typique (erreur médiane) : du {(typical + pd.Timedelta(days=1)):%d/%m/%Y}")
    fig_semaine(bt, worst, "pire_semaine",
                f"La pire semaine : du {(worst + pd.Timedelta(days=1)):%d/%m/%Y} (rupture brutale du niveau)")
    fig_prevision(y, fc)

    cols = ["date", "origine", "horizon", "modele", "famille", "reel", "prevision", "borne_basse", "borne_haute"]
    bt[cols].round(2).to_csv(REPORTS / "previsions_backtest.csv", index=False)

    def records(df):
        return json.loads(df.round(3).to_json(orient="records", force_ascii=False))

    report = {
        "date": date.today().isoformat(),
        "donnees": {"jours": len(y), "debut": str(y.index[0].date()), "fin": str(y.index[-1].date()),
                    "moyenne_gwh_jour": round(float(y.mean()), 1)},
        "protocole": {"periode_test": [TEST_START, TEST_END], "semaines": len(origins), "horizon_jours": 7,
                      "intervalle": "80 %", "echelle_mase_gwh": round(scale, 2),
                      "combinaison": COMBINED_MEMBERS},
        "modele_retenu": best,
        "semaine_typique": str((typical + pd.Timedelta(days=1)).date()),
        "pire_semaine": {"debut": str((worst + pd.Timedelta(days=1)).date()), "mae_gwh": round(float(werr.iloc[-1]), 1)},
        "resultats": records(table),
        "par_annee": records(by_year),
        "par_saison": records(by_season),
        "par_horizon": records(by_h),
        "tests_diebold_mariano": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in tests],
        "temps_calcul_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRésultats (104 semaines de test) :")
    print(table[["modele", "mae", "mape", "mase", "biais", "couverture_80"]].round(2).to_string(index=False))
    print("\nTests de Diebold-Mariano (le meilleur contre chacun) :")
    for t in tests:
        verdict = "écart significatif" if t["p_value"] < 0.05 else "écart NON significatif"
        print(f"  {t['modele_a']} vs {t['modele_b']:22s} p = {t['p_value']:.3f}  → {verdict}")
    print(f"\nFigures : {FIG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapide", action="store_true", help="26 semaines seulement")
    main(parser.parse_args().rapide)
