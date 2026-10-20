"""
Modele inferencyjne dla hipotezy o statusie rodzinnym a ryzyku upadlosci.

Zawiera:
  - firth_logit / firth_profile_ci : regresja logistyczna Firtha (1993) z
    penalizacja Jeffreysa - odporna na rzadkie zdarzenia i quasi-separacje
    (King & Zeng, 2001). Zaimplementowana bez zaleznosci zewnetrznych.
  - firth_family_model           : model Firtha dla 'family' z kontrolami,
    z profilowym PL CI oraz permutacyjnym p-value klastrowanym po firmie.
  - hierarchical_hazard_logit    : hierarchiczny logit hazardu M1->M3
    (pooled logit, bledy standardowe klastrowane po firmie).
  - raw_rate_by_family           : surowa stopa upadlosci + test chi2.
  - stratified_rates             : stopa upadlosci w warstwach (kompozycja).
  - nace_section                 : mapowanie dzialu NACE -> sekcja (litera).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

# ============================================================
# 1. REGRESJA FIRTHA (penalizowana wiarygodnosc, Jeffreys prior)
# ============================================================

def _pen_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray):
    eta = np.clip(X @ beta, -500, 500)
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    W = p * (1 - p)
    I = (X * W[:, None]).T @ X
    sign, logdet = np.linalg.slogdet(I)
    if sign <= 0:
        logdet = -1e10
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * logdet)
    return ll, p, W, I


def firth_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 200, tol: float = 1e-8,
                init: np.ndarray | None = None) -> dict:
    """Regresja logistyczna Firtha. X musi zawierac kolumne sta, jesli chcemy intercept.

    init: opcjonalny punkt startowy (warm-start) - przyspiesza zbieznosc np. w
    tescie permutacyjnym, gdzie kolejne dopasowania roznia sie tylko kolumna family.

    Zwraca dict: beta, se, pval (Wald, chi2 z 1 st. sw.), loglik, n, converged.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    beta = np.zeros(k) if init is None else np.array(init, float)
    ll_old = -np.inf
    converged = False
    for _ in range(max_iter):
        ll, p, W, I = _pen_loglik(beta, X, y)
        Iinv = np.linalg.pinv(I)
        XI = X @ Iinv
        h = W * np.einsum('ij,ij->i', XI, X)            # diag macierzy kapelusza
        U = X.T @ (y - p + h * (0.5 - p))               # zmodyfikowany score
        step = Iinv @ U
        f = 1.0                                          # step-halving dla stabilnosci
        ll_new = ll
        while f > 1e-6:
            ll_try, *_ = _pen_loglik(beta + f * step, X, y)
            if ll_try >= ll - 1e-10:
                ll_new = ll_try
                break
            f *= 0.5
        beta = beta + f * step
        if abs(ll_new - ll_old) < tol:
            converged = True
            ll_old = ll_new
            break
        ll_old = ll_new
    ll, p, W, I = _pen_loglik(beta, X, y)
    Iinv = np.linalg.pinv(I)
    se = np.sqrt(np.clip(np.diag(Iinv), 0, np.inf))
    with np.errstate(divide='ignore', invalid='ignore'):
        wald = np.where(se > 0, (beta / se) ** 2, 0.0)
    pval = stats.chi2.sf(wald, 1)
    return dict(beta=beta, se=se, pval=pval, loglik=ll, n=n, converged=converged)


def _profile_loglik(bk: float, X: np.ndarray, y: np.ndarray, idx: int) -> float:
    """Maks. penalizowana wiarygodnosc przy ustalonym beta[idx] = bk."""
    cols = [j for j in range(X.shape[1]) if j != idx]
    Xr = X[:, cols]
    offset = X[:, idx] * bk
    beta = np.zeros(Xr.shape[1])
    for _ in range(200):
        eta = np.clip(Xr @ beta + offset, -500, 500)
        p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        I = (Xr * W[:, None]).T @ Xr
        Iinv = np.linalg.pinv(I)
        XI = Xr @ Iinv
        h = W * np.einsum('ij,ij->i', XI, Xr)
        U = Xr.T @ (y - p + h * (0.5 - p))
        step = Iinv @ U
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    bfull = np.zeros(X.shape[1])
    bfull[idx] = bk
    bfull[cols] = beta
    ll, *_ = _pen_loglik(bfull, X, y)
    return ll


def firth_profile_ci(X: np.ndarray, y: np.ndarray, idx: int, alpha: float = 0.05,
                     full: dict | None = None, max_span: float = 100.0):
    """Profilowy przedzial ufnosci (penalized likelihood) dla wspolczynnika idx.

    Zwraca (lo, hi); wartosc +/- inf gdy przedzial jest nieograniczony.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if full is None:
        full = firth_logit(X, y)
    ll_max = full['loglik']
    bhat = full['beta'][idx]
    se = max(full['se'][idx], 0.1)
    crit = stats.chi2.ppf(1 - alpha, 1)

    def g(bk):
        return 2 * (ll_max - _profile_loglik(bk, X, y, idx)) - crit

    def find_bound(direction):
        step = se
        prev = bhat
        cur = bhat + direction * step
        while g(cur) < 0:
            step *= 1.7
            prev = cur
            cur = bhat + direction * step
            if abs(cur - bhat) > max_span:
                return direction * np.inf
        lo_b, hi_b = (prev, cur) if direction > 0 else (cur, prev)
        return brentq(g, lo_b, hi_b, xtol=1e-4)

    return find_bound(-1), find_bound(+1)


# ============================================================
# 2. MODEL FIRTHA DLA 'family' Z KONTROLAMI + TEST PERMUTACYJNY
# ============================================================

def firth_family_model(df: pd.DataFrame, target: str, controls: list[str],
                       group_col: str = 'TIN', n_perm: int = 0, seed: int = 42,
                       perm_cap_iter: int = 6, verbose: bool = True) -> dict:
    """Firth dla efektu 'family' z kontrolami liczbowymi.

    CZESC SZYBKA (zawsze): OR, profilowe PL CI oraz p-Walda dla 'family' -
    na pelnym panelu ~1.5 mln wierszy liczy sie w kilka minut.

    CZESC WOLNA (opcjonalna, n_perm>0): permutacyjne p-value klastrowane po firmie
    (status rodzinny tasowany na poziomie firmy). Kazda permutacja to osobne
    dopasowanie Firtha na calym panelu, dlatego uzywamy WARM-STARTU (start z
    dopasowanych kontrol, family=0) i limitu iteracji. Szacunkowo ~1-2 s na
    permutacje przy 1.5 mln wierszy -> n_perm=1000 to ~15-25 min. Zacznij od
    n_perm=0 (tylko OR+CI+Wald), a permutacje odpal swiadomie gdy masz czas.

    'controls' musza byc kolumnami liczbowymi bez brakow (wczesniej przygotowanymi).
    """
    import time

    use = [target, 'family', group_col] + controls
    d = df[use].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = d[target].to_numpy(float)
    Xcols = ['family'] + controls
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in Xcols])
    names = ['const'] + Xcols
    fam_idx = names.index('family')

    if verbose:
        print(f"Firth: dopasowanie na {len(d):,} wierszach, {int(y.sum())} zdarzen "
              f"(rodzinnych: {int(d.loc[d['family'] == 1, target].sum())})...")
    t0 = time.time()
    fit = firth_logit(X, y)
    lo, hi = firth_profile_ci(X, y, idx=fam_idx, full=fit)
    or_fam = float(np.exp(fit['beta'][fam_idx]))
    if verbose:
        print(f"  -> OR={or_fam:.3f}, PL CI [{np.exp(lo):.3f}, {np.exp(hi):.3f}], "
              f"p_wald={fit['pval'][fam_idx]:.3g}  ({time.time() - t0:.1f}s)")

    perm_p = float('nan')
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        firm_family = d.groupby(group_col)['family'].first()
        firms = firm_family.index.to_numpy()
        fam_by_firm = firm_family.to_numpy(float)
        pos = {f: i for i, f in enumerate(firms)}
        row_firm_pos = np.array([pos[f] for f in d[group_col].to_numpy()])

        obs_stat = abs(fit['beta'][fam_idx])
        warm = fit['beta'].copy()
        warm[fam_idx] = 0.0                      # kontrole z dopasowania, family w H0
        ge = 0
        tp = time.time()
        for i in range(n_perm):
            Xp = X.copy()
            Xp[:, fam_idx] = rng.permutation(fam_by_firm)[row_firm_pos]
            fp = firth_logit(Xp, y, max_iter=perm_cap_iter, init=warm)
            if abs(fp['beta'][fam_idx]) >= obs_stat - 1e-12:
                ge += 1
            if verbose and (i + 1) % max(1, n_perm // 10) == 0:
                el = time.time() - tp
                eta = el / (i + 1) * (n_perm - i - 1)
                print(f"  permutacje {i + 1}/{n_perm}  (elapsed {el/60:.1f} min, "
                      f"ETA {eta/60:.1f} min)")
        perm_p = (ge + 1) / (n_perm + 1)
        if verbose:
            print(f"  -> p_perm={perm_p:.4f}  ({(time.time() - tp)/60:.1f} min)")

    coefs = pd.DataFrame({
        'coef': fit['beta'],
        'OR': np.exp(fit['beta']),
        'se': fit['se'],
        'p_wald': fit['pval'],
    }, index=names)
    return dict(
        table=coefs.round(4),
        family_OR=or_fam,
        family_CI=(float(np.exp(lo)), float(np.exp(hi))),
        family_p_wald=float(fit['pval'][fam_idx]),
        family_p_perm=float(perm_p),
        n=int(fit['n']),
        n_events=int(y.sum()),
        n_events_family=int(d.loc[d['family'] == 1, target].sum()),
        converged=fit['converged'],
    )


# ============================================================
# 3. HIERARCHICZNY LOGIT HAZARDU (M1 -> M3), SE KLASTROWANE PO FIRMIE
# ============================================================

def nace_section(divcode) -> str:
    """Dzial NACE (2 cyfry) -> sekcja (litera). 'X' gdy nie da sie przypisac."""
    bins = [(1, 3, 'A'), (5, 9, 'B'), (10, 33, 'C'), (35, 35, 'D'), (36, 39, 'E'),
            (41, 43, 'F'), (45, 47, 'G'), (49, 53, 'H'), (55, 56, 'I'), (58, 63, 'J'),
            (64, 66, 'K'), (68, 68, 'L'), (69, 75, 'M'), (77, 82, 'N'), (84, 84, 'O'),
            (85, 85, 'P'), (86, 88, 'Q'), (90, 93, 'R'), (94, 96, 'S')]
    try:
        dv = int(divcode)
    except (ValueError, TypeError):
        return 'X'
    for lo, hi, s in bins:
        if lo <= dv <= hi:
            return s
    return 'X'


def _prepare_inference_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Kontrole IDENTYCZNE jak w firth_family_model (CTRL = log_assets, firm_age,
    # zero_assets) -> M2 to dokladnie ta sama specyfikacja co Firth, tylko estymowana
    # metoda najwiekszej wiarygodnosci. Wlasnosc swiadomie pominieta (separacja).
    d = df.copy()
    d['log_assets'] = np.log(d['TotalAssets'].clip(lower=1))
    d['nace_sec'] = d['nace_div'].apply(nace_section)
    d['yearc'] = d['year'].astype(int).astype(str)
    keep = [target, 'family', 'log_assets', 'firm_age', 'zero_assets', 'nace_sec',
            'yearc', 'TIN']
    return d[keep].replace([np.inf, -np.inf], np.nan).dropna()


def hierarchical_hazard_logit(df: pd.DataFrame, target: str = 'will_fail_next_year') -> dict:
    """Estymuje M1 (family), M2 (+rozmiar/wiek/wlasnosc), M3 (+sektor+rok).

    SE klastrowane po firmie (TIN). Zwraca df_used oraz tabele OR family per model.
    """
    import statsmodels.formula.api as smf

    d = _prepare_inference_frame(df, target)

    def fit(formula):
        try:
            return smf.logit(formula, data=d).fit(disp=False, method='newton',
                                                   cov_type='cluster',
                                                   cov_kwds={'groups': d['TIN']})
        except Exception:
            return smf.logit(formula, data=d).fit(disp=False, method='bfgs', maxiter=300)

    specs = {
        'M1: tylko family': f'{target} ~ family',
        'M2: + rozmiar/wiek/status': f'{target} ~ family + log_assets + firm_age + zero_assets',
        'M3: + sektor + rok': f'{target} ~ family + log_assets + firm_age + zero_assets + C(nace_sec) + C(yearc)',
    }
    rows = []
    fitted = {}
    for name, f in specs.items():
        try:
            m = fit(f)
            fitted[name] = m
            orf = float(np.exp(m.params['family']))
            lo, hi = np.exp(m.conf_int().loc['family'])
            rows.append(dict(model=name, family_OR=orf, CI_lo=float(lo), CI_hi=float(hi),
                             p=float(m.pvalues['family']), n_obs=int(m.nobs)))
        except Exception as e:
            rows.append(dict(model=name, family_OR=np.nan, CI_lo=np.nan, CI_hi=np.nan,
                             p=np.nan, n_obs=np.nan, error=type(e).__name__))
    return dict(
        table=pd.DataFrame(rows).set_index('model'),
        n=len(d),
        n_events=int(d[target].sum()),
        n_events_family=int(d.loc[d['family'] == 1, target].sum()),
        fitted=fitted,
        frame=d,
    )


# ============================================================
# 4. STATYSTYKI OPISOWE / NIEPARAMETRYCZNE
# ============================================================

def raw_rate_by_family(df: pd.DataFrame, target: str = 'will_fail_next_year') -> dict:
    """Surowa stopa upadlosci wg family + test niezaleznosci chi2."""
    r = df.groupby('family')[target].agg(['mean', 'sum', 'count'])
    r.columns = ['stopa_upadlosci', 'liczba_upadlosci', 'firmo_lata']
    r.index = ['Nierodzinne (0)', 'Rodzinne (1)']
    tab = pd.crosstab(df['family'], df[target])
    chi2, pval, _, _ = stats.chi2_contingency(tab)
    denom = r.loc['Rodzinne (1)', 'stopa_upadlosci']
    ratio = (r.loc['Nierodzinne (0)', 'stopa_upadlosci'] / denom) if denom > 0 else np.inf
    return dict(table=r, chi2=float(chi2), p=float(pval), ratio=float(ratio))


def stratified_rates(df: pd.DataFrame, by: str, target: str = 'will_fail_next_year',
                     n_quantiles: int | None = None, top_n: int | None = None) -> pd.DataFrame:
    """Stopa upadlosci (%) wg family w warstwach zmiennej 'by'.

    n_quantiles: podzial na kwantyle (np. wielkosc). top_n: TOP-N kategorii (np. NACE).
    """
    d = df.copy()
    if n_quantiles is not None:
        labels = [f'Q{i+1}' for i in range(n_quantiles)]
        d['_strat'] = pd.qcut(d[by].rank(method='first'), n_quantiles, labels=labels)
        idx = '_strat'
    else:
        if top_n is not None:
            top = d[by].value_counts().nlargest(top_n).index
            d = d[d[by].isin(top)]
        idx = by
    rate = d.pivot_table(target, index=idx, columns='family', aggfunc='mean',
                         observed=True) * 100
    rate.columns = ['Nierodzinne %', 'Rodzinne %'][:rate.shape[1]]
    cnt = d.pivot_table(target, index=idx, columns='family', aggfunc='sum', observed=True)
    cnt.columns = ['n_upad_0', 'n_upad_1'][:cnt.shape[1]]
    return rate.join(cnt).round(4)
