"""
Wspolna infrastruktura testow.

Testy NIE dotykaja data/raw/dane.dta - ten plik wazy 3,48 GB, jest licencjonowany
i nie ma go w repozytorium. Zamiast tego kazdy test dostaje maly panel syntetyczny
o tej samej STRUKTURZE co dane Orbis: jeden wiersz = firmo-rok, zmienna liczba lat
na firme, czesc firm ze zdarzeniem upadlosci, braki w czesci wskaznikow.

Logika, ktorej pilnuja testy (podzial po firmach, brak wyciekl czasowego,
winsoryzacja tylko na treningu), nie zalezy od rozmiaru proby - 60 firm
wystarczy, zeby ja sprawdzic, a testy wykonuja sie w sekundy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

RATIO_COLS = [f'X{i}' for i in range(1, 6)]


def zbuduj_panel(n_firm: int = 60, min_lat: int = 2, max_lat: int = 7,
                 n_zdarzen: int = 12, rok_od: int = 2013, rok_do: int = 2020,
                 braki: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Syntetyczny panel firmo-lat o strukturze zgodnej z danymi zrodlowymi.

    n_zdarzen firm dostaje rok upadlosci mieszczacy sie w ich oknie obserwacji,
    pozostale maja NaN. Kolumna yearofbankruptcy jest - tak jak w Orbisie -
    powtorzona w kazdym wierszu danej firmy.
    """
    rng = np.random.default_rng(seed)
    wiersze = []
    tiny = 1_000_000 + np.arange(n_firm)
    upadle = set(rng.choice(tiny, size=min(n_zdarzen, n_firm), replace=False))

    for tin in tiny:
        lat = int(rng.integers(min_lat, max_lat + 1))
        start = int(rng.integers(rok_od, rok_do - lat + 2))
        lata = list(range(start, start + lat))

        rok_upadlosci = np.nan
        if tin in upadle and lat >= 2:
            # zdarzenie w oknie obserwacji, ale nie w pierwszym roku
            rok_upadlosci = float(lata[int(rng.integers(1, lat))])

        for rok in lata:
            wiersze.append({
                'TIN': float(tin),
                'year': float(rok),
                'companyname': f'FIRMA {tin}',
                'NACECode': float(rng.integers(1000, 9999)),
                'TotalAssets': float(rng.lognormal(10, 2)),
                'Sales': float(rng.lognormal(10, 2)),
                'family': float(rng.random() < 0.3),
                'n_corp_group': float(rng.integers(0, 5)),
                'SHAREHOLDERS': float(rng.integers(1, 20)),
                'n_employees_': float(rng.integers(1, 500)),
                'yearofbankruptcy': rok_upadlosci,
                'yearofliquidation': np.nan,
                **{c: float(rng.normal()) for c in RATIO_COLS},
            })

    df = pd.DataFrame(wiersze)

    # Braki w czesci wskaznikow - nielosowe braki to istota tych danych
    maska = rng.random((len(df), len(RATIO_COLS))) < braki
    for j, c in enumerate(RATIO_COLS):
        df.loc[maska[:, j], c] = np.nan

    return df.sort_values(['TIN', 'year']).reset_index(drop=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    """Surowy panel, przed jakimkolwiek przetworzeniem."""
    return zbuduj_panel()


@pytest.fixture
def panel_z_targetem(panel) -> pd.DataFrame:
    """Panel po zbudowaniu zmiennej celu (model hazardu)."""
    from pre_processing import build_hazard_target
    return build_hazard_target(panel, ['yearofbankruptcy'])


@pytest.fixture
def panel_po_podziale(panel_z_targetem) -> pd.DataFrame:
    """Panel z cechami panelowymi i podzialem na train/val/test."""
    from pre_processing import add_panel_features, split_firms
    df = add_panel_features(panel_z_targetem.copy(), RATIO_COLS)
    return split_firms(df, test_pct=0.20, val_pct=0.12, seed=42)
