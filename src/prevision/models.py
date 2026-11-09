"""
Modèles de prévision comparés. Tous ont la même interface :

    modele.fit(y)              y : consommation journalière (GWh) jusqu'au jour de la prévision
    modele.predict(h)          -> DataFrame indexé par les h jours suivants,
                                  colonnes : prevision, borne_basse, borne_haute (intervalle à 80 %)
"""

from __future__ import annotations

import logging
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import calendar_features

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

ALPHA = 0.2  # intervalle à 80 %
Z80 = 1.2816  # quantile 90 % de la loi normale


def future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")


def _frame(idx, mean, low, high) -> pd.DataFrame:
    return pd.DataFrame({"prevision": mean, "borne_basse": low, "borne_haute": high}, index=idx)


class Forecaster:
    name = "?"
    family = "?"

    def fit(self, y: pd.Series) -> "Forecaster":
        raise NotImplementedError

    def update(self, y: pd.Series) -> "Forecaster":
        """Ajoute les nouvelles observations sans réestimer les paramètres (par défaut : réestime)."""
        return self.fit(y)

    def predict(self, h: int) -> pd.DataFrame:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# 1. Références naïves : le minimum à battre
# ─────────────────────────────────────────────────────────────
class SeasonalNaive(Forecaster):
    """Chaque jour = le même jour de la semaine précédente."""

    name, family = "Naïf saisonnier", "Référence"

    def fit(self, y):
        self.y = y
        err = (y - y.shift(7)).dropna().iloc[-730:]  # erreurs des 2 dernières années
        self.q_low, self.q_high = np.quantile(err, [ALPHA / 2, 1 - ALPHA / 2])
        return self

    def predict(self, h):
        idx = future_index(self.y, h)
        last_week = self.y.iloc[-7:].to_numpy()
        mean = np.array([last_week[i % 7] for i in range(h)])
        return _frame(idx, mean, mean + self.q_low, mean + self.q_high)


class LastYear(Forecaster):
    """Chaque jour = le même jour de la semaine, un an plus tôt (364 jours), corrigé du niveau récent."""

    name, family = "Naïf annuel ajusté", "Référence"

    def fit(self, y):
        self.y = y
        ratio = (y.rolling(28).mean() / y.shift(364).rolling(28).mean())
        pred = y.shift(364) * ratio.shift(7)
        err = (y - pred).dropna().iloc[-730:]
        self.q_low, self.q_high = np.quantile(err, [ALPHA / 2, 1 - ALPHA / 2])
        self.ratio = y.iloc[-28:].mean() / y.iloc[-28 - 364:-364].mean()
        return self

    def predict(self, h):
        idx = future_index(self.y, h)
        mean = self.y.reindex(idx - pd.Timedelta(days=364)).to_numpy() * self.ratio
        return _frame(idx, mean, mean + self.q_low, mean + self.q_high)


# ─────────────────────────────────────────────────────────────
# 2. Modèles statistiques classiques
# ─────────────────────────────────────────────────────────────
class ETS(Forecaster):
    """Lissage exponentiel (Holt-Winters) : niveau + saisonnalité hebdomadaire, sur le log."""

    name, family = "ETS (Holt-Winters)", "Statistique"

    def __init__(self, history_days: int = 3 * 365):
        self.history_days = history_days

    def fit(self, y):
        self.y = y
        z = np.log(y.iloc[-self.history_days:])
        self.res = ETSModel(z, error="add", trend=None, seasonal="add", seasonal_periods=7).fit(disp=False)
        return self

    def predict(self, h):
        idx = future_index(self.y, h)
        sims = self.res.simulate(h, repetitions=500, anchor="end")
        sims = np.exp(np.asarray(sims))
        mean = np.exp(np.asarray(self.res.forecast(h)))
        low, high = np.quantile(sims, [ALPHA / 2, 1 - ALPHA / 2], axis=1)
        return _frame(idx, mean, low, high)


class SarimaxFourier(Forecaster):
    """SARIMA hebdomadaire + termes de Fourier (saison annuelle) + jours fériés, sur le log."""

    name, family = "SARIMAX + Fourier", "Statistique"
    EXOG = ["ferie", "pont", "fetes_fin_annee", "aout",
            "annee_sin1", "annee_cos1", "annee_sin2", "annee_cos2", "annee_sin3", "annee_cos3"]

    def __init__(self, history_days: int = 4 * 365):
        self.history_days = history_days

    def fit(self, y):
        self.y = y
        z = np.log(y.iloc[-self.history_days:])
        exog = calendar_features(z.index)[self.EXOG]
        self.res = SARIMAX(z, exog=exog, order=(2, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
        return self

    def update(self, y):
        self.y = y
        z = np.log(y.iloc[-self.history_days:])
        self.res = self.res.apply(z, exog=calendar_features(z.index)[self.EXOG], refit=False)
        return self

    def predict(self, h):
        idx = future_index(self.y, h)
        fc = self.res.get_forecast(h, exog=calendar_features(idx)[self.EXOG])
        ci = fc.conf_int(alpha=ALPHA)
        return _frame(idx, np.exp(fc.predicted_mean.to_numpy()),
                      np.exp(ci.iloc[:, 0].to_numpy()), np.exp(ci.iloc[:, 1].to_numpy()))


class ProphetModel(Forecaster):
    """Prophet (Meta) : tendance par morceaux + saisons annuelle et hebdomadaire + jours fériés français."""

    name, family = "Prophet", "Statistique"

    def __init__(self, history_days: int = 4 * 365):
        self.history_days = history_days

    def fit(self, y):
        from prophet import Prophet
        logging.getLogger("cmdstanpy").disabled = True  # messages techniques de Stan
        self.y = y
        z = y.iloc[-self.history_days:]
        df = pd.DataFrame({"ds": z.index, "y": np.log(z.to_numpy())})
        self.m = Prophet(yearly_seasonality=10, weekly_seasonality=True, daily_seasonality=False,
                         interval_width=1 - ALPHA, seasonality_mode="additive", changepoint_prior_scale=0.1)
        self.m.add_country_holidays(country_name="FR")
        self.m.fit(df)
        return self

    def predict(self, h):
        idx = future_index(self.y, h)
        fc = self.m.predict(pd.DataFrame({"ds": idx}))
        return _frame(idx, np.exp(fc["yhat"].to_numpy()),
                      np.exp(fc["yhat_lower"].to_numpy()), np.exp(fc["yhat_upper"].to_numpy()))


# ─────────────────────────────────────────────────────────────
# 3. Machine learning : LightGBM, stratégie directe
# ─────────────────────────────────────────────────────────────
def ml_features(y: pd.Series, origins: pd.DatetimeIndex, h_max: int) -> pd.DataFrame:
    """
    Une ligne par (jour d'origine t, horizon h) : on prévoit le jour T = t + h
    en n'utilisant QUE ce qui est connu au soir du jour t.
    La cible est le rapport entre la consommation du jour T et le niveau récent (moyenne des 7 derniers jours) :
    un arbre ne sait pas extrapoler un niveau jamais vu, mais il sait prévoir un rapport.
    """
    rows = []
    ylog = np.log(y)
    level = ylog.rolling(7).mean()
    for h in range(1, h_max + 1):
        T = origins + pd.Timedelta(days=h)
        f = calendar_features(T)
        f.index = origins
        f["horizon"] = h
        base = level.reindex(origins).to_numpy()
        f["niveau_7j"] = base
        f["dernier_jour"] = ylog.reindex(origins).to_numpy() - base
        f["tendance_7j"] = base - level.reindex(origins - pd.Timedelta(days=7)).to_numpy()
        f["niveau_28j"] = ylog.rolling(28).mean().reindex(origins).to_numpy() - base
        # même jour de la semaine, 1 et 2 semaines avant T (toujours connus car h ≤ 7)
        f["lag_semaine"] = ylog.reindex(T - pd.Timedelta(days=7)).to_numpy() - base
        f["lag_2semaines"] = ylog.reindex(T - pd.Timedelta(days=14)).to_numpy() - base
        # l'an dernier : même jour de la semaine, et niveau de l'époque
        f["lag_an"] = ylog.reindex(T - pd.Timedelta(days=364)).to_numpy() - level.reindex(origins - pd.Timedelta(days=364)).to_numpy()
        f["cible"] = ylog.reindex(T).to_numpy() - base
        f["date_cible"] = T
        rows.append(f)
    return pd.concat(rows)


class LightGBMDirect(Forecaster):
    """
    LightGBM en stratégie directe : un seul modèle pour les 7 horizons (l'horizon est une variable).
    Intervalles : régression quantile, puis calibration conforme (CQR) sur la dernière année.
    """

    name, family = "LightGBM", "Machine learning"
    PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    CALIBRATION_DAYS = 365

    def __init__(self, h_max: int = 7, history_days: int = 8 * 365, conformal: bool = True):
        self.h_max, self.history_days, self.conformal = h_max, history_days, conformal

    def _dataset(self, y):
        z = y.iloc[-self.history_days:]
        origins = z.index[400:-1]  # il faut un an d'historique pour les variables « l'an dernier »
        data = ml_features(y, origins, self.h_max).dropna(subset=["cible"])
        return data[data["date_cible"] <= y.index[-1]]

    def _fit_models(self, data):
        X, t = data[self.cols], data["cible"]
        return (lgb.LGBMRegressor(objective="regression", **self.PARAMS).fit(X, t),
                lgb.LGBMRegressor(objective="quantile", alpha=ALPHA / 2, **self.PARAMS).fit(X, t),
                lgb.LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, **self.PARAMS).fit(X, t))

    def fit(self, y):
        self.y = y
        data = self._dataset(y)
        self.cols = [c for c in data.columns if c not in ("cible", "date_cible", "niveau_7j")]
        self.margin = np.zeros(self.h_max)
        if self.conformal:
            # 1) on entraîne sans la dernière année, 2) on mesure de combien les intervalles ratent la cible
            #    sur cette année jamais vue, 3) on élargit les intervalles d'autant, horizon par horizon
            cut = y.index[-1] - pd.Timedelta(days=self.CALIBRATION_DAYS)
            train, calib = data[data["date_cible"] <= cut], data[data.index > cut]
            _, low_m, high_m = self._fit_models(train)
            score = np.maximum(low_m.predict(calib[self.cols]) - calib["cible"],
                               calib["cible"] - high_m.predict(calib[self.cols]))
            for h in range(1, self.h_max + 1):
                s = score[calib["horizon"] == h].to_numpy()
                level = min(1.0, np.ceil((len(s) + 1) * (1 - ALPHA)) / len(s))
                self.margin[h - 1] = np.quantile(s, level)
        self.mean_model, self.low_model, self.high_model = self._fit_models(data)
        return self

    def update(self, y):
        self.y = y  # mêmes arbres, nouvelles valeurs récentes
        return self

    def predict(self, h):
        idx = future_index(self.y, h)
        f = ml_features(self.y, pd.DatetimeIndex([self.y.index[-1]]), h)
        base = f["niveau_7j"].to_numpy()
        X = f[self.cols]
        margin = self.margin[:h]
        mean = np.exp(base + self.mean_model.predict(X))
        low = np.exp(base + self.low_model.predict(X) - margin)
        high = np.exp(base + self.high_model.predict(X) + margin)
        return _frame(idx, mean, np.minimum(low, mean), np.maximum(high, mean))


def all_models() -> list[Forecaster]:
    return [SeasonalNaive(), LastYear(), ETS(), SarimaxFourier(), ProphetModel(), LightGBMDirect()]
