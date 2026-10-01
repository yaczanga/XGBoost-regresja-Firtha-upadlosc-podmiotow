"""
Testy gwarancji metodologicznych przygotowania danych.

Kazdy test odpowiada jednemu twierdzeniu z rozdzialu metodologicznego pracy.
Wszystkie dzialaja na panelu syntetycznym - patrz conftest.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import RATIO_COLS, zbuduj_panel
from pre_processing import (add_panel_features, build_hazard_target,
                            build_model_features, clean_panel, make_xy,
                            split_firms, winsorize_train_only)


# ============================================================
# Zmienna celu: model hazardu w czasie dyskretnym
# ============================================================

def test_target_oznacza_wylacznie_rok_poprzedzajacy_zdarzenie(panel_z_targetem):
    """Kazda obserwacja pozytywna pochodzi z roku bezposrednio przed upadloscia.

    To jest zabezpieczenie przed wyciekiem czasowym: gdyby flaga upadlosci byla
    zapalona przez wiele lat tej samej firmy, model rozpoznawalby stan juz
    trwajacy zamiast przewidywac przyszle zdarzenie.
    """
    zdarzenia = panel_z_targetem[panel_z_targetem['will_fail_next_year'] == 1]
    assert len(zdarzenia) > 0, 'panel testowy nie zawiera zadnego zdarzenia'
    assert (zdarzenia['year'] + 1 == zdarzenia['event_year']).all()


def test_brak_obserwacji_od_roku_zdarzenia_wzwyz(panel_z_targetem):
    """Od ogloszenia upadlosci podmiot nie jest juz narazony na ryzyko."""
    ze_zdarzeniem = panel_z_targetem[panel_z_targetem['event_year'].notna()]
    assert (ze_zdarzeniem['year'] < ze_zdarzeniem['event_year']).all()


def test_firma_wnosi_najwyzej_jedna_obserwacje_pozytywna(panel_z_targetem):
    """Przy rocznym horyzoncie kazda firma moze upasc tylko raz."""
    na_firme = panel_z_targetem.groupby('TIN')['will_fail_next_year'].sum()
    assert (na_firme <= 1).all()


def test_cenzurowanie_prawostronne_usuwa_ostatni_rok_firm_bez_zdarzenia(panel):
    """Dla firmy bez zdarzenia nie znamy stanu w roku nastepnym po ostatnim
    obserwowanym, wiec ten rok nie moze wnosic obserwacji negatywnej."""
    przed = panel[panel['yearofbankruptcy'].isna()].groupby('TIN')['year'].max()
    po = build_hazard_target(panel.copy(), ['yearofbankruptcy'])
    po_bez = po[po['event_year'].isna()].groupby('TIN')['year'].max()

    wspolne = przed.index.intersection(po_bez.index)
    assert len(wspolne) > 0
    assert (po_bez[wspolne] < przed[wspolne]).all()


# ============================================================
# Czyszczenie panelu
# ============================================================

def test_clean_panel_usuwa_wiersze_powyzej_progu_brakow(panel):
    df = clean_panel(panel.copy(), RATIO_COLS, max_missing=1)
    assert (df[RATIO_COLS].isnull().sum(axis=1) <= 1).all()


def test_clean_panel_flaguje_niedodatnie_aktywa_zamiast_usuwac(panel):
    """Podmioty o zerowych aktywach to 39% zdarzen - wykluczenie ich usuneloby
    znaczna czesc modelowanego zjawiska, wiec dostaja flage."""
    df = panel.copy()
    df.loc[df.index[:5], 'TotalAssets'] = 0.0
    df.loc[df.index[5:8], 'TotalAssets'] = -100.0
    wynik = clean_panel(df, RATIO_COLS, max_missing=len(RATIO_COLS))

    assert len(wynik) == len(df), 'zaden wiersz nie powinien zostac usuniety'
    assert wynik['zero_assets'].sum() == 8
    assert (wynik.loc[wynik['TotalAssets'] > 0, 'zero_assets'] == 0).all()


# ============================================================
# Cechy panelowe
# ============================================================

def test_add_panel_features_tworzy_dokladnie_oczekiwane_kolumny(panel_z_targetem):
    przed = set(panel_z_targetem.columns)
    df = add_panel_features(panel_z_targetem.copy(), RATIO_COLS)
    nowe = set(df.columns) - przed

    assert nowe == {f'{c}_yoy' for c in RATIO_COLS} | {'firm_age', 'nace_div'}
    # kolumny z poprzedniej wersji projektu, ktorych funkcja juz nie tworzy
    assert 'ebitda_margin' not in df.columns
    assert 'tangibility' not in df.columns


def test_zmiany_rok_do_roku_nie_przeciekaja_miedzy_firmami():
    """Pierwszy obserwowany rok kazdej firmy nie ma poprzednika, wiec jego
    zmiana musi byc brakiem - inaczej diff zlapalby wartosc innej firmy."""
    df = zbuduj_panel(n_firm=10, braki=0.0, seed=1)
    df = add_panel_features(df, RATIO_COLS)
    pierwszy_rok = df.groupby('TIN')['year'].transform('min') == df['year']
    assert df.loc[pierwszy_rok, 'X1_yoy'].isna().all()
    assert df.loc[~pierwszy_rok, 'X1_yoy'].notna().all()


def test_build_model_features_nie_zawiera_roku_kalendarzowego():
    """'year' jest celowo pomijany: zdarzenia z likwidacji datowane na ostatni
    rok firmy tworzylyby sztuczne skupienie."""
    feats, delty = build_model_features(['family'], RATIO_COLS)
    assert 'year' not in feats
    assert delty == [f'{c}_yoy' for c in RATIO_COLS]
    assert len(feats) == len(set(feats)), 'cechy nie moga sie powtarzac'


# ============================================================
# Podzial proby
# ============================================================

def test_zadna_firma_nie_trafia_do_dwoch_zbiorow(panel_po_podziale):
    """Podzial na poziomie firm, nie wierszy. Przy podziale po wierszach rozne
    lata tej samej spolki trafilyby jednoczesnie do uczenia i testu, a model
    rozpoznawalby konkretne podmioty zamiast ogolnych prawidlowosci."""
    assert (panel_po_podziale.groupby('TIN')['split'].nunique() == 1).all()


def test_podzial_pokrywa_wszystkie_wiersze(panel_po_podziale):
    assert panel_po_podziale['split'].isin({'train', 'val', 'test'}).all()
    assert panel_po_podziale['split'].notna().all()


def test_podzial_jest_powtarzalny_dla_tego_samego_ziarna(panel_z_targetem):
    a = split_firms(panel_z_targetem.copy(), 0.20, 0.12, seed=42)['split']
    b = split_firms(panel_z_targetem.copy(), 0.20, 0.12, seed=42)['split']
    c = split_firms(panel_z_targetem.copy(), 0.20, 0.12, seed=7)['split']

    assert a.equals(b)
    assert not a.equals(c), 'inne ziarno powinno dac inny podzial'


# ============================================================
# Winsoryzacja
# ============================================================

def test_progi_winsoryzacji_licza_sie_wylacznie_na_treningu(panel_po_podziale):
    """Progi policzone na calosci przenosilyby informacje ze zbioru
    walidacyjnego i testowego do przygotowania danych.

    Sprawdzamy to wstawiajac skrajna wartosc TYLKO poza trening: wiersze
    treningowe musza wyjsc identycznie jak bez niej.
    """
    bazowy = panel_po_podziale.copy()
    bazowy['X1'] = bazowy['X1'].fillna(0.0)

    skazony = bazowy.copy()
    idx_test = skazony.index[skazony['split'] == 'test'][0]
    skazony.loc[idx_test, 'X1'] = 1e9

    a = winsorize_train_only(bazowy.copy(), ['X1'], clip=0.01)
    b = winsorize_train_only(skazony, ['X1'], clip=0.01)

    maska = a['split'] == 'train'
    pd.testing.assert_series_equal(a.loc[maska, 'X1'], b.loc[maska, 'X1'])
    # skrajna wartosc musi zostac przyciecta do progu z treningu
    assert b.loc[idx_test, 'X1'] == pytest.approx(a.loc[maska, 'X1'].max())


def test_winsoryzacja_pomija_kolumny_spoza_ramki(panel_po_podziale):
    """Brak kolumny nie moze wysadzic potoku."""
    wynik = winsorize_train_only(panel_po_podziale.copy(), ['X1', 'nie_ma_takiej'])
    assert 'nie_ma_takiej' not in wynik.columns


# ============================================================
# Macierze modelu
# ============================================================

def test_make_xy_zwraca_spojne_ksztalty_i_grupy(panel_po_podziale):
    df = winsorize_train_only(panel_po_podziale.copy(), RATIO_COLS)
    feats = RATIO_COLS + ['firm_age', 'nace_div']
    X, y, groups, ids = make_xy(df, feats, 'train', ['TIN', 'year', 'companyname'])

    assert X.shape == (len(y), len(feats))
    assert len(groups) == len(y)
    assert len(ids) == len(y)
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert set(groups) == set(df.loc[df['split'] == 'train', 'TIN'])


def test_make_xy_nie_miesza_zbiorow(panel_po_podziale):
    df = panel_po_podziale
    _, _, g_train, _ = make_xy(df, RATIO_COLS, 'train', ['TIN'])
    _, _, g_test, _ = make_xy(df, RATIO_COLS, 'test', ['TIN'])
    assert not (set(g_train) & set(g_test))
