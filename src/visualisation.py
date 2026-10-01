"""
Funkcje wizualizacyjne dla notebooka (wykresy EDA, oceny modelu i inferencji).
Kazda funkcja tworzy i pokazuje figure; zwraca obiekt Axes/Figure gdy przydatny.

Zapis do plikow. Domyslnie wykresy tylko sie wyswietlaja. Po wywolaniu
set_figures_dir(sciezka) kazdy wykres jest dodatkowo zapisywany do pliku PNG
pod nazwa podana w argumencie 'nazwa'. Sluzy to do odtworzenia zawartosci
reports/figures/, ktora trafia do repozytorium i do README.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

FIGURES_DIR: Path | None = None


def set_figures_dir(path: str | Path | None) -> None:
    """Ustawia katalog zapisu wykresow (None wylacza zapis)."""
    global FIGURES_DIR
    FIGURES_DIR = None if path is None else Path(path)
    if FIGURES_DIR is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Wykresy beda zapisywane do: {FIGURES_DIR}")


def _finish(nazwa: str | None) -> None:
    """Zapisuje biezaca figure (gdy ustawiono katalog) i ja wyswietla."""
    if FIGURES_DIR is not None and nazwa:
        plt.savefig(FIGURES_DIR / f'{nazwa}.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_current(nazwa: str) -> None:
    """Zapisuje i wyswietla figure utworzona poza tym modulem - np. wykres SHAP,
    rysowany bezposrednio w notebooku przez biblioteke shap."""
    _finish(nazwa)


def plot_target_distribution(df: pd.DataFrame, target: str = 'will_fail_next_year',
                             nazwa: str | None = '01_rozklad_zmiennej_celu'):
    dist = df[target].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Zdrowe (0)', 'Upadlosc (1)'],
                y=[dist.get(0.0, 0), dist.get(1.0, 0)],
                hue=['Zdrowe (0)', 'Upadlosc (1)'],
                palette=['#2ecc71', '#e74c3c'], legend=False)
    plt.title('Rozklad zmiennej celu (will_fail_next_year)')
    plt.ylabel('Liczba obserwacji (skala log)')
    plt.yscale('log')
    plt.tight_layout()
    _finish(nazwa)


def plot_corr_matrix(df: pd.DataFrame, cols: list[str],
                     nazwa: str | None = '02_macierz_korelacji_wskaznikow'):
    plt.figure(figsize=(14, 10))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, annot_kws={'size': 7})
    plt.title('Macierz korelacji wskaznikow finansowych')
    plt.tight_layout()
    _finish(nazwa)


def plot_confusion(y_true, y_pred, threshold: float,
                   nazwa: str | None = '03_macierz_pomylek'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Zdrowa (0)', 'Upadlosc (1)'],
                yticklabels=['Zdrowa (0)', 'Upadlosc (1)'])
    plt.ylabel('Rzeczywistosc'); plt.xlabel('Predykcja')
    plt.title(f'Macierz pomylek (prog={threshold:.3f}, test)')
    plt.tight_layout()
    _finish(nazwa)


def plot_pr_curve(y_true, y_score, pr_auc: float | None = None,
                  nazwa: str | None = '04_krzywa_precision_recall'):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(8, 5))
    label = f'PR AUC = {pr_auc:.4f}' if pr_auc is not None else 'PR'
    plt.plot(rec, prec, color='darkorange', lw=2, label=label)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Krzywa Precision-Recall (test)')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(nazwa)


def plot_learning_curve(results: dict, metric_key: str = 'aucpr',
                        nazwa: str | None = '05_krzywa_uczenia'):
    if not results:
        return
    key = metric_key if metric_key in results['validation_0'] else list(results['validation_0'])[0]
    epochs = range(len(results['validation_0'][key]))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['validation_0'][key], label='Trening')
    if 'validation_1' in results:
        plt.plot(epochs, results['validation_1'][key], label='Walidacja')
    plt.title(f'Krzywa uczenia ({key}) - XGBoost')
    plt.xlabel('Liczba drzew'); plt.ylabel(key.upper())
    plt.legend(); plt.tight_layout()
    _finish(nazwa)


def plot_failure_rate_by_size(df: pd.DataFrame, target: str = 'will_fail_next_year',
                              size_col: str = 'TotalAssets',
                              nazwa: str | None = '09_stopa_upadlosci_wg_wielkosci_i_typu'):
    d = df.copy()
    d['size_q'] = pd.qcut(d[size_col].rank(method='first'), 4,
                          labels=['Q1 (male)', 'Q2', 'Q3', 'Q4 (duze)'])
    g = (d.groupby(['size_q', 'family'], observed=True)[target]
         .mean().mul(100).reset_index())
    g['Typ'] = g['family'].map({0.0: 'Nierodzinne', 1.0: 'Rodzinne'})
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=g, x='size_q', y=target, hue='Typ',
                     palette={'Nierodzinne': '#e74c3c', 'Rodzinne': '#2ecc71'})
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f%%', fontsize=8, padding=2)
    plt.title('Stopa upadlosci wg wielkosci firmy: rodzinne vs nierodzinne')
    plt.xlabel('Kwartyl wielkosci (aktywa ogolem)')
    plt.ylabel('Stopa upadlosci (%)')
    plt.legend(title='Typ firmy')
    plt.tight_layout()
    _finish(nazwa)


def plot_risk_by_family(analiza_df: pd.DataFrame,
                        score_col: str = 'Ryzyko_Upadlosci_Score',
                        nazwa: str | None = '08_rozklad_ryzyka_rodzinne_vs_nierodzinne'):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='family', y=score_col, hue='family', data=analiza_df,
                palette='Set2', legend=False)
    plt.title('Rozklad score ryzyka: Nierodzinne (0) vs Rodzinne (1)')
    plt.xlabel('Typ firmy (0=Nierodzinna, 1=Rodzinna)')
    plt.ylabel('Score ryzyka (XGBoost, nieskalibrowany)')
    plt.tight_layout()
    _finish(nazwa)
