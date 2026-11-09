"""Chargement de la consommation électrique française et calendrier (jours fériés, ponts)."""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "consommation_rte_30min.csv"


def load_halfhourly(path: Path = RAW) -> pd.DataFrame:
    """Consommation nationale au pas de 30 minutes, en MW (source : RTE éCO2mix)."""
    df = pd.read_csv(path)
    df["horodatage"] = pd.to_datetime(df["date"] + " " + df["heure"])
    return df[["horodatage", "consommation_mw"]]


def load_daily(path: Path = RAW) -> pd.Series:
    """Énergie consommée chaque jour, en GWh (somme des puissances × 0,5 h)."""
    df = load_halfhourly(path)
    daily = df.groupby(df["horodatage"].dt.normalize())["consommation_mw"].sum() * 0.5 / 1000
    daily.index = pd.DatetimeIndex(daily.index, freq="D", name="date")
    daily.name = "consommation_gwh"
    return daily


def french_holidays(start: str = "2011-01-01", end: str = "2026-12-31") -> pd.Series:
    """Jours fériés en France métropolitaine (nom du jour férié, indexé par date)."""
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    fr = holidays.France(years=years)
    s = pd.Series({pd.Timestamp(d): name for d, name in fr.items()}).sort_index()
    return s.loc[start:end]


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendrier connues à l'avance (utilisables pour prévoir le futur)."""
    dates = pd.DatetimeIndex(dates)
    ferie = dates.isin(french_holidays().index)
    # pont : jour ouvré coincé entre un jour férié et le week-end (lundi ou vendredi)
    veille_ferie = (dates + pd.Timedelta(days=1)).isin(french_holidays().index)
    lendemain_ferie = (dates - pd.Timedelta(days=1)).isin(french_holidays().index)
    pont = ((dates.dayofweek == 0) & veille_ferie) | ((dates.dayofweek == 4) & lendemain_ferie)
    doy = dates.dayofyear.to_numpy()
    feats = pd.DataFrame(index=dates)
    feats["jour_semaine"] = dates.dayofweek
    feats["mois"] = dates.month
    feats["ferie"] = ferie.astype(int)
    feats["pont"] = pont.astype(int)
    # période de Noël : consommation industrielle très basse
    feats["fetes_fin_annee"] = (((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 1))).astype(int)
    feats["aout"] = (dates.month == 8).astype(int)
    for k in (1, 2, 3):
        feats[f"annee_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        feats[f"annee_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return feats
