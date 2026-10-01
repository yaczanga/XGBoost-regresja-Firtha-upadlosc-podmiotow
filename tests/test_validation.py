"""
Testy doboru progu decyzyjnego.

Prog dobierany jest tak, by osiagnac zadana czulosc. Przy przenoszeniu go
miedzy modelami kluczowa jest obserwacja, ze predykcje out-of-fold pochodza
z kilku modeli o roznych skalach score'ow - dlatego przenosi sie odsetek
alarmow, a nie wartosc bezwzgledna progu.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation import (alarm_rate_from_oof, threshold_from_alarm_rate,
                        threshold_from_scores)


@pytest.fixture
def rzadkie_zdarzenia():
    """2000 obserwacji, 40 zdarzen, score skorelowany z klasa, ale slabo."""
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros(n)
    y[rng.choice(n, size=40, replace=False)] = 1.0
    scores = rng.random(n) * 0.6
    scores[y == 1] += rng.random(int(y.sum())) * 0.4
    return y, scores


def test_prog_osiaga_zadana_czulosc(rzadkie_zdarzenia):
    y, scores = rzadkie_zdarzenia
    for cel in (0.80, 0.90, 0.95):
        thr = threshold_from_scores(y, scores, target_recall=cel)
        recall = (scores[y == 1] >= thr).mean()
        assert recall >= cel


def test_wyzsza_zadana_czulosc_daje_nie_wyzszy_prog(rzadkie_zdarzenia):
    """Zeby zlapac wiecej zdarzen, trzeba obnizyc poprzeczke."""
    y, scores = rzadkie_zdarzenia
    progi = [threshold_from_scores(y, scores, c) for c in (0.70, 0.80, 0.90, 0.95)]
    assert progi == sorted(progi, reverse=True)


def test_brak_osiagalnego_progu_daje_wartosc_domyslna():
    """Gdy zadanej czulosci nie da sie osiagnac, funkcja zwraca 0.5 zamiast
    wysypywac sie na pustej masce."""
    y = np.array([0.0, 0.0, 1.0])
    scores = np.array([0.9, 0.8, 0.1])
    assert threshold_from_scores(y, scores, target_recall=1.01) == 0.5


def test_odsetek_alarmow_jest_niezmienniczy_na_skale_score(rzadkie_zdarzenia):
    """Sedno przenoszenia progu miedzy modelami.

    Monotoniczne przeskalowanie score'ow (tu: podniesienie do potegi) nie moze
    zmienic ani liczby alarmow, ani uzyskanej czulosci - inaczej prog z
    walidacji krzyzowej nie przenosilby sie na model koncowy.
    """
    y, scores = rzadkie_zdarzenia
    rate = alarm_rate_from_oof(y, scores, target_recall=0.90)

    for przeskalowanie in (lambda s: s ** 2.5, lambda s: np.sqrt(s), lambda s: s * 100):
        inne = przeskalowanie(scores)
        thr = threshold_from_alarm_rate(inne, rate)
        pred = inne > thr

        assert pred.sum() == pytest.approx(rate * len(y), abs=2)
        assert pred[y == 1].mean() >= 0.85


def test_odsetek_alarmow_miesci_sie_w_zakresie(rzadkie_zdarzenia):
    y, scores = rzadkie_zdarzenia
    rate = alarm_rate_from_oof(y, scores, target_recall=0.90)
    assert 0.0 < rate <= 1.0
