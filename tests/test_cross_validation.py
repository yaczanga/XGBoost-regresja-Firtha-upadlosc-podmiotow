"""
Testy integralnosci podzialu na foldy w walidacji krzyzowej.

Walidacja krzyzowa na panelu firmo-lat ma dwa wymogi: foldy nie moga dzielic
firm (lata tej samej spolki sa skorelowane) i musza miec zblizona liczbe
zdarzen (przy jednym zdarzeniu na kilka tysiecy wierszy zwykly GroupKFold
potrafi wygenerowac fold bez ani jednego zdarzenia, co czyni srednia z foldow
bezuzyteczna).
"""
from __future__ import annotations

import numpy as np
import pytest

from cross_validation import make_folds


def dane_grupowe(n_firm: int = 300, lat: int = 5, n_zdarzen: int = 40,
                 seed: int = 0):
    """Grupy i zmienna celu o strukturze panelu: zdarzenie najwyzej raz na firme."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_firm), lat)
    y = np.zeros(len(groups))
    for f in rng.choice(n_firm, size=n_zdarzen, replace=False):
        idx = np.where(groups == f)[0]
        y[idx[rng.integers(len(idx))]] = 1.0
    return y, groups


@pytest.fixture
def dane():
    return dane_grupowe()


def test_zadna_firma_nie_wystepuje_po_obu_stronach_foldu(dane):
    y, groups = dane
    for tr, va in make_folds(y, groups, n_splits=5, seed=42):
        assert not (set(groups[tr]) & set(groups[va]))


def test_kazda_obserwacja_dokladnie_raz_w_czesci_walidacyjnej(dane):
    y, groups = dane
    pokrycie = np.zeros(len(y), dtype=int)
    for _, va in make_folds(y, groups, n_splits=5, seed=42):
        pokrycie[va] += 1
    assert (pokrycie == 1).all()


def test_czesci_uczaca_i_walidacyjna_sa_rozlaczne_i_pelne(dane):
    y, groups = dane
    for tr, va in make_folds(y, groups, n_splits=5, seed=42):
        assert not (set(tr) & set(va))
        assert len(tr) + len(va) == len(y)


def test_zdarzenia_rozlozone_rownomiernie_miedzy_foldy(dane):
    """Stratyfikacja po zmiennej celu. Przy 40 zdarzeniach i 5 foldach kazdy
    powinien dostac okolo 8 - dopuszczamy odchylenie o jedno zdarzenie."""
    y, groups = dane
    na_fold = [int(y[va].sum()) for _, va in make_folds(y, groups, 5, seed=42)]

    assert sum(na_fold) == int(y.sum())
    assert min(na_fold) > 0, 'fold bez zdarzen czyni srednia bezuzyteczna'
    assert max(na_fold) - min(na_fold) <= 1


def test_podzial_na_foldy_jest_powtarzalny(dane):
    y, groups = dane
    a = make_folds(y, groups, n_splits=5, seed=42)
    b = make_folds(y, groups, n_splits=5, seed=42)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


@pytest.mark.parametrize('n_splits', [3, 5, 10])
def test_liczba_foldow_jest_respektowana(dane, n_splits):
    y, groups = dane
    assert len(make_folds(y, groups, n_splits=n_splits, seed=42)) == n_splits
