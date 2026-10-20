"""
Ocena modelu predykcyjnego: prog decyzyjny, klastrowany bootstrap metryk,
test zrodla sygnalu (proxy 'martwej firmy' vs wskazniki finansowe).
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)


def threshold_from_scores(y: np.ndarray, scores: np.ndarray,
                          target_recall: float) -> float:
    """Najwyzszy prog dajacy recall >= target_recall (0.5 gdy taki nie istnieje)."""
    prec, rec, thr = precision_recall_curve(y, scores)
    mask = rec[:-1] >= target_recall              # rec/prec maja dlugosc L, thr -> L-1
    return float(thr[mask][-1]) if mask.any() else 0.5


def find_threshold(model: "xgb.XGBClassifier", X_val: np.ndarray, y_val: np.ndarray,
                   target_recall: float) -> float:
    """Najwyzszy prog na WALIDACJI dajacy recall >= target_recall (0.5 gdy brak)."""
    threshold = threshold_from_scores(y_val, model.predict_proba(X_val)[:, 1],
                                      target_recall)
    print(f"Prog z walidacji dla recall>={target_recall:.0%}: {threshold:.4f}")
    return threshold


def alarm_rate_from_oof(y: np.ndarray, oof_scores: np.ndarray,
                        target_recall: float) -> float:
    """Jaki odsetek firmo-lat trzeba oznaczyc alarmem, by zlapac target_recall zdarzen.

    Wielkosc ta jest niezmiennicza wzgledem monotonicznego przeskalowania
    score'ow, wiec przenosi sie miedzy modelami o roznej kalibracji - inaczej
    niz prog w wartosciach bezwzglednych.
    """
    thr = threshold_from_scores(y, oof_scores, target_recall)
    rate = float((oof_scores > thr).mean())
    print(f"Odsetek alarmow potrzebny dla recall>={target_recall:.0%}: "
          f"{rate:.4%}  (na {int(y.sum())} zdarzeniach out-of-fold)")
    return rate


def threshold_from_alarm_rate(scores: np.ndarray, alarm_rate: float) -> float:
    """Prog odpowiadajacy zadanemu odsetkowi alarmow w rozkladzie score'ow modelu."""
    return float(np.quantile(scores, 1.0 - alarm_rate))


def bootstrap_ci(y_test: np.ndarray, y_pred: np.ndarray, threshold: float,
                 groups: np.ndarray | None = None, B: int = 2000,
                 seed: int = 42) -> dict[str, tuple[float, float, float]]:
    """Bootstrapowe przedzialy ufnosci metryk testowych.

    UWAGA metodologiczna: przy panelu firmo-lat losujemy ZE ZWRACANIEM CALE FIRMY
    (klastry TIN), a nie pojedyncze wiersze - inaczej CI sa zanizone, bo wiersze
    tej samej firmy sa skorelowane. Podaj 'groups' (tablica TIN dla kazdego wiersza).
    Gdy groups=None -> klasyczny bootstrap po wierszach (tylko dla i.i.d.).
    """
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    n = len(y_test)

    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        # mapowanie firma -> indeksy jej wierszy (dla szybkiego skladania probek)
        idx_by_group = {g: np.where(groups == g)[0] for g in uniq}

    prauc_bs, roc_bs, recall_bs, prec_bs = [], [], [], []
    for _ in range(B):
        if groups is not None:
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in sampled])
        else:
            idx = rng.integers(0, n, n)
        yt_b, yp_b = y_test[idx], y_pred[idx]
        if yt_b.sum() == 0:
            continue
        prauc_bs.append(average_precision_score(yt_b, yp_b))
        roc_bs.append(roc_auc_score(yt_b, yp_b))
        pred_b = (yp_b > threshold).astype(int)
        pos = yt_b == 1
        recall_bs.append(pred_b[pos].mean())
        prec_bs.append(pred_b[pos].sum() / pred_b.sum() if pred_b.sum() > 0 else 0.0)

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5))

    return {"PR-AUC": ci(prauc_bs), "ROC-AUC": ci(roc_bs),
            "Recall@prog": ci(recall_bs), "Precision@prog": ci(prec_bs)}


def signal_source_test(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_features: list[str], ratio_cols: list[str],
                       scale_weight: float,
                       proxy_feats: tuple[str, ...] = ('zero_assets', 'missing_count', 'firm_age')
                       ) -> dict[str, float]:
    """Ile mocy predykcyjnej pochodzi z proxy 'martwej firmy' vs samych wskaznikow.

    Trenuje 3 szybkie modele XGBoost (jednakowe hiperparametry) na:
      - wszystkich cechach,
      - cechach BEZ proxy (zero_assets, missing_count, firm_age),
      - TYLKO wskaznikach finansowych (ratio_cols).
    Zwraca PR-AUC (test) dla kazdego wariantu.
    """
    idx_all = list(range(len(model_features)))
    idx_noproxy = [i for i, f in enumerate(model_features) if f not in proxy_feats]
    idx_onlyratios = [i for i, f in enumerate(model_features) if f in ratio_cols]

    quick = dict(objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                 scale_pos_weight=scale_weight, random_state=42, n_jobs=-1,
                 n_estimators=1500, early_stopping_rounds=50,
                 learning_rate=0.05, max_depth=5)

    def prauc(cols_idx):
        m = xgb.XGBClassifier(**quick)
        m.fit(X_train[:, cols_idx], y_train,
              eval_set=[(X_val[:, cols_idx], y_val)], verbose=False)
        p = m.predict_proba(X_test[:, cols_idx])[:, 1]
        return float(average_precision_score(y_test, p))

    return {
        'wszystkie': prauc(idx_all),
        'bez_proxy': prauc(idx_noproxy),
        'tylko_wskazniki': prauc(idx_onlyratios),
    }
