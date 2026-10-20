"""
Walidacja krzyzowa z podzialem na poziomie firm.

Podzial idzie po firmach (grupa = TIN) i ze stratyfikacja po zmiennej celu, co
razem spelnia StratifiedGroupKFold: rozne lata tej samej spolki sa skorelowane,
a przy jednym zdarzeniu na 3540 firmo-lat zwykly GroupKFold daje foldy o
skrajnie roznej liczbie zdarzen. Zbior testowy nie bierze udzialu w tej
procedurze na zadnym etapie.
"""
from __future__ import annotations

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold


# 1. PODZIAL NA FOLDY

def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
               seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Foldy z podzialem na poziomie firm i stratyfikacja po zmiennej celu.

    Zwraca liste par (indeksy_uczace, indeksy_walidacyjne).
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def folds_report(y: np.ndarray, groups: np.ndarray,
                 folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Diagnostyka podzialu: licznosci, zdarzenia i kontrola rozlacznosci firm."""
    print(f"{'Fold':>5} {'Wierszy ucz.':>13} {'Wierszy wal.':>13} "
          f"{'Zdarzen wal.':>13} {'Firm wal.':>11}")
    for k, (tr, va) in enumerate(folds, 1):
        print(f"{k:>5} {len(tr):>13,} {len(va):>13,} "
              f"{int(y[va].sum()):>13} {len(np.unique(groups[va])):>11,}")

    # Kontrola: zadna firma nie moze wystapic po obu stronach podzialu
    przecieki = [k for k, (tr, va) in enumerate(folds, 1)
                 if set(np.unique(groups[tr])) & set(np.unique(groups[va]))]
    print(f"\nFoldy z ta sama firma po obu stronach (ma byc puste): {przecieki}")

    pokrycie = np.zeros(len(y), dtype=int)
    for _, va in folds:
        pokrycie[va] += 1
    print(f"Kazda obserwacja dokladnie raz w czesci walidacyjnej: "
          f"{bool((pokrycie == 1).all())}")


# 2. OCENA ZESTAWU HIPERPARAMETROW NA FOLDACH

FIXED = dict(
    objective='binary:logistic',
    eval_metric='aucpr',
    n_jobs=-1,
    random_state=42,
    tree_method='hist',
    n_estimators=3000,
    early_stopping_rounds=50,
)


def cv_evaluate(params: dict, X: np.ndarray, y: np.ndarray,
                folds: list[tuple[np.ndarray, np.ndarray]],
                trial: "optuna.Trial | None" = None) -> dict:
    """PR-AUC na kazdym foldzie + predykcje out-of-fold.

    scale_pos_weight liczony jest OSOBNO dla kazdego foldu, z jego wlasnej
    czesci uczacej - inaczej do wagi klasy przeciekalaby informacja o rozkladzie
    zdarzen w czesci walidacyjnej.

    Czesc walidacyjna foldu sluzy zarowno wczesnemu zatrzymaniu, jak i ocenie,
    wiec wynik jest nieznacznie optymistyczny. Jest to standardowy kompromis:
    alternatywa (zagniezdzony podzial na wczesne zatrzymanie) zwielokrotnia czas
    obliczen, a ocena koncowa i tak odbywa sie na zbiorze testowym, ktory w
    walidacji krzyzowej nie uczestniczy.

    trial: gdy podany, po kazdym foldzie raportuje wynik do Optuny, co pozwala
    przyciac beznadziejne proby bez liczenia pozostalych foldow.
    """
    oof = np.full(len(y), np.nan)
    wyniki, drzewa = [], []

    for k, (tr, va) in enumerate(folds):
        pos = float(y[tr].sum())
        neg = len(tr) - pos
        p = dict(FIXED, **params,
                 scale_pos_weight=neg / pos if pos > 0 else 1.0)

        model = xgb.XGBClassifier(**p)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

        pred = model.predict_proba(X[va])[:, 1]
        oof[va] = pred
        wyniki.append(float(average_precision_score(y[va], pred)))

        bi = getattr(model, 'best_iteration', None)
        drzewa.append(int(bi) + 1 if bi is not None else p['n_estimators'])

        if trial is not None:
            trial.report(float(np.mean(wyniki)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # prauc_oof bywa nizsze niz prauc_mean i nie jest to blad: predykcje
    # pochodza z roznych modeli, kazdy o wlasnej skali score'ow, wiec ich
    # zlaczenie miesza rozklady. Do strojenia sluzy prauc_mean; prauc_oof jest
    # miara pomocnicza, a prog decyzyjny przenosi sie przez odsetek alarmow
    # (validation.alarm_rate_from_oof), nie przez wartosc bezwzgledna progu.
    return {
        'prauc_folds': wyniki,
        'prauc_mean': float(np.mean(wyniki)),
        'prauc_std': float(np.std(wyniki, ddof=1)) if len(wyniki) > 1 else 0.0,
        'prauc_oof': float(average_precision_score(y, oof)),
        'oof': oof,
        'n_trees': drzewa,
    }


# 3. STROJENIE HIPERPARAMETROW NA WALIDACJI KRZYZOWEJ

SEARCH_SPACE = {
    'learning_rate':    ('float', 0.01, 0.3, True),
    'max_depth':        ('int', 2, 5, False),
    'min_child_weight': ('int', 10, 100, False),
    'subsample':        ('float', 0.4, 0.8, False),
    'colsample_bytree': ('float', 0.4, 0.8, False),
    'gamma':            ('float', 0.5, 10.0, False),
    'reg_alpha':        ('float', 0.1, 50.0, True),
    'reg_lambda':       ('float', 1.0, 100.0, True),
    'max_delta_step':   ('int', 0, 10, False),
}


def _suggest(trial: "optuna.Trial") -> dict:
    """Ta sama przestrzen przeszukiwania co w przebiegu bez walidacji krzyzowej."""
    out = {}
    for nazwa, (typ, lo, hi, log) in SEARCH_SPACE.items():
        if typ == 'int':
            out[nazwa] = trial.suggest_int(nazwa, lo, hi)
        else:
            out[nazwa] = trial.suggest_float(nazwa, lo, hi, log=log)
    return out


def train_model_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_trials: int, storage: str, study_name: str,
                   n_splits: int = 5, seed: int = 42):
    """Strojenie XGBoost z walidacja krzyzowa na poziomie firm.

    X, y, groups obejmuja POLACZONY zbior uczacy i walidacyjny. Zbior testowy
    pozostaje nietkniety.

    Funkcja celu to srednia PR-AUC po foldach. Zastosowano pruner medianowy,
    ktory przerywa proby odstajace in minus po pierwszych foldach - bez tego
    koszt bylby n_splits razy wiekszy niz przy pojedynczym podziale.

    Model koncowy douczany jest na calym X z liczba drzew rowna medianie z
    foldow. Nie ma juz wydzielonego zbioru do wczesnego zatrzymania, a mediana
    jest na nie odporniejsza niz srednia.

    Zwraca (model, study, wynik_cv), gdzie wynik_cv zawiera m.in. predykcje
    out-of-fold do doboru progu decyzyjnego.
    """
    folds = make_folds(y, groups, n_splits=n_splits, seed=seed)
    print(f"Podzial na {n_splits} foldow (poziom firm, stratyfikacja po zmiennej celu):")
    folds_report(y, groups, folds)
    print()

    def objective(trial):
        return cv_evaluate(_suggest(trial), X, y, folds, trial=trial)['prauc_mean']

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    zrobione = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    zostalo = max(0, n_trials - zrobione)
    print(f"Prob zakonczonych w bazie: {zrobione}. Do wykonania teraz: {zostalo}.")
    if zostalo > 0:
        study.optimize(objective, n_trials=zostalo, show_progress_bar=True)
    else:
        print("Osiagnieto zadana liczbe prob - zwieksz n_trials, by stroic dalej.")

    print(f"\nNajlepsza srednia PR-AUC z foldow: {study.best_value:.4f}")
    print(f"Najlepsze parametry: {study.best_params}")

    # Przeliczenie najlepszego zestawu raz jeszcze, by miec rozrzut i predykcje OOF
    wynik = cv_evaluate(study.best_params, X, y, folds)
    print(f"PR-AUC na foldach: "
          f"{', '.join(f'{v:.4f}' for v in wynik['prauc_folds'])}")
    print(f"Srednia {wynik['prauc_mean']:.4f} +/- {wynik['prauc_std']:.4f} (odch. std.)")
    print(f"PR-AUC na polaczonych predykcjach out-of-fold: {wynik['prauc_oof']:.4f}")
    print(f"Drzew po wczesnym zatrzymaniu w foldach: {wynik['n_trees']}")

    n_trees = int(np.median(wynik['n_trees']))
    pos = float(y.sum())
    neg = len(y) - pos
    final = dict(FIXED, **study.best_params)
    final.pop('early_stopping_rounds')
    final['n_estimators'] = n_trees
    final['scale_pos_weight'] = neg / pos if pos > 0 else 1.0

    print(f"\nDouczanie modelu koncowego na calym zbiorze uczacym "
          f"({len(y):,} wierszy, {n_trees} drzew)...")
    model = xgb.XGBClassifier(**final)
    model.fit(X, y, verbose=False)

    return model, study, wynik
