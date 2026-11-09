import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prevision.backtest import diebold_mariano, mase_scale, metrics, run_backtest, weekly_origins
from prevision.data import calendar_features, load_daily
from prevision.models import ETS, LastYear, SeasonalNaive, ml_features

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def y():
    return load_daily()


def test_donnees_completes(y):
    assert len(y) == 4383  # 12 années, sans trou
    assert y.index.is_monotonic_increasing and y.index.freq == "D"
    assert y.isna().sum() == 0
    assert 800 < y.min() and y.max() < 2400  # GWh par jour : ordre de grandeur de la France


def test_calendrier():
    f = calendar_features(pd.to_datetime(["2023-07-14", "2023-05-19", "2023-05-17", "2023-12-25"]))
    assert f.loc["2023-07-14", "ferie"] == 1  # fête nationale
    assert f.loc["2023-05-19", "pont"] == 1   # vendredi après l'Ascension
    assert f.loc["2023-05-17", ["ferie", "pont"]].sum() == 0
    assert f.loc["2023-12-25", "fetes_fin_annee"] == 1


def test_naif_saisonnier(y):
    train = y.loc[:"2023-06-04"]
    fc = SeasonalNaive().fit(train).predict(7)
    assert fc.index[0] == pd.Timestamp("2023-06-05")
    np.testing.assert_allclose(fc["prevision"].to_numpy(), train.iloc[-7:].to_numpy())
    assert (fc["borne_basse"] <= fc["prevision"]).all() and (fc["prevision"] <= fc["borne_haute"]).all()


def test_variables_ml_sans_fuite_du_futur(y):
    """Les variables d'une prévision faite le jour t ne doivent dépendre que du passé (≤ t)."""
    origin = pd.Timestamp("2023-03-05")
    f1 = ml_features(y, pd.DatetimeIndex([origin]), 7)
    y2 = y.copy()
    y2.loc[origin + pd.Timedelta(days=1):] *= 3  # on falsifie complètement le futur
    f2 = ml_features(y2, pd.DatetimeIndex([origin]), 7)
    cols = [c for c in f1.columns if c not in ("cible", "date_cible")]
    pd.testing.assert_frame_equal(f1[cols], f2[cols])
    assert not np.allclose(f1["cible"], f2["cible"])


def test_backtest_et_metriques(y):
    origins = weekly_origins(y, "2023-01-01", "2023-02-28")
    bt = run_backtest(y, [SeasonalNaive(), LastYear(), ETS()], origins, verbose=False)
    assert set(bt["horizon"]) == set(range(1, 8))
    assert bt.groupby("modele").size().nunique() == 1
    # aucune prévision n'utilise de donnée postérieure à son origine
    assert (bt["date"] > bt["origine"]).all()
    m = metrics(bt[bt["modele"] == "ETS (Holt-Winters)"], mase_scale(y, "2022-12-31"))
    assert 0 < m["mape"] < 20 and 0 <= m["couverture_80"] <= 100


def test_diebold_mariano_symetrique(y):
    origins = weekly_origins(y, "2023-01-01", "2023-06-30")
    bt = run_backtest(y, [SeasonalNaive(), ETS()], origins, verbose=False)
    a = diebold_mariano(bt, "ETS (Holt-Winters)", "Naïf saisonnier")
    b = diebold_mariano(bt, "Naïf saisonnier", "ETS (Holt-Winters)")
    assert a["p_value"] == pytest.approx(b["p_value"])
    assert a["statistique"] == pytest.approx(-b["statistique"])


def test_rapport_genere():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("lancer d'abord : cd src && python -m prevision.run")
    report = json.loads(path.read_text(encoding="utf-8"))
    models = {r["modele"] for r in report["resultats"]}
    assert len(models) == 7
    assert report["modele_retenu"] in models
    for fig in ("serie", "comparaison", "erreur_horizon", "couverture", "exemple_semaine"):
        assert (ROOT / "reports" / "figures" / f"{fig}.png").exists()
