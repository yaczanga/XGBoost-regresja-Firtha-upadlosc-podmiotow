"""
Funkcje wizualizacyjne dla notebooka (wykresy EDA, oceny modelu i inferencji).
Kazda funkcja tworzy i pokazuje figure; zwraca obiekt Axes/Figure gdy przydatny.

Barwy. Zaden wykres nie dobiera koloru samodzielnie - wszystkie siegaja po
tokeny z modulu plot_style, opisujace ROLE koloru w danych. Zasada generujaca
systemu brzmi: cynober to przedmiot uwagi, blekit to odniesienie. Wywolanie
style.apply_style() ustawia typografie, siatke i ramki.

Jezyk. Tekst widoczny na rysunku jest po polsku, z diakrytykami i przecinkiem
dziesietnym. Nazwy metryk takze tlumaczymy (precyzja, czulosc, pole pod krzywa
precyzja-czulosc). Komentarze i docstringi sa po polsku bez ogonkow, a nazwy
w kodzie po angielsku - zgodnie z reszta projektu.

Zapis do plikow. Domyslnie wykresy tylko sie wyswietlaja. Po wywolaniu
set_figures_dir(sciezka) kazdy wykres jest dodatkowo zapisywany do pliku PNG
pod nazwa podana w argumencie 'name'. Sluzy to do odtworzenia zawartosci
reports/figures/, ktora trafia do repozytorium i do README.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

import plot_style as style

# Styl ustawiamy przy imporcie. Modul istnieje wylacznie po to, by rysowac
# wykresy tego projektu, wiec nie ma stanu, w ktorym mialby rysowac w stylu
# domyslnym matplotliba - a jawne wywolanie w notebooku bywaloby pomijane.
style.apply_style()

FIGURES_DIR: Path | None = None


def _pl(value: float, places: int = 2) -> str:
    """Liczba w zapisie polskim - przecinek dziesietny."""
    return f'{value:.{places}f}'.replace('.', ',')


def _thousands(value) -> str:
    """Liczba calkowita w zapisie polskim.

    Separator tysiecy stosujemy dopiero od pieciu cyfr - tak jak w tekscie
    projektu: "2072 obserwacje", ale "1 317 608 firmo-lat". Separatorem jest
    spacja nierozdzielajaca, zeby liczba nie lamala sie miedzy grupami.
    """
    number = int(value)
    if abs(number) < 10000:
        return str(number)
    return f'{number:,}'.replace(',', ' ')


def set_figures_dir(path: str | Path | None) -> None:
    """Ustawia katalog zapisu wykresow (None wylacza zapis)."""
    global FIGURES_DIR
    FIGURES_DIR = None if path is None else Path(path)
    if FIGURES_DIR is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Wykresy beda zapisywane do: {FIGURES_DIR}")


def _finish(name: str | None) -> None:
    """Zapisuje biezaca figure (gdy ustawiono katalog) i ja wyswietla."""
    if FIGURES_DIR is not None and name:
        plt.savefig(FIGURES_DIR / f'{name}.png', dpi=style.SAVE_DPI,
                    bbox_inches='tight')
    plt.show()


def save_current(name: str) -> None:
    """Zapisuje i wyswietla figure utworzona poza tym modulem - np. wykres SHAP,
    rysowany bezposrednio w notebooku przez biblioteke shap."""
    _finish(name)


def plot_target_distribution(df: pd.DataFrame, target: str = 'will_fail_next_year',
                             name: str | None = '01_rozklad_zmiennej_celu'):
    dist = df[target].value_counts()
    fig, ax = plt.subplots(figsize=style.SMALL)
    ax.bar(['Firmy zdrowe (0)', 'Upadłość (1)'],
           [dist.get(0.0, 0), dist.get(1.0, 0)],
           color=[style.REFERENCE, style.RISK], width=0.55)
    ax.set_yscale('log')
    ax.set_ylabel('Liczba obserwacji (skala logarytmiczna)')
    ax.set_title('Rozkład zmiennej celu')
    for container in ax.containers:
        texts = [_thousands(v) for v in container.datavalues]
        ax.bar_label(container, labels=texts, fontsize=9, padding=3,
                     color=style.TEXT_SECONDARY)
    plt.tight_layout()
    _finish(name)


def plot_corr_matrix(df: pd.DataFrame, cols: list[str],
                     name: str | None = '02_macierz_korelacji_wskaznikow'):
    # Przekatna zawsze wynosi 1,00, a oba trojkaty niosa te sama informacje.
    # Bez maski najmocniejsza barwa systemu trafia na komorki bez tresci.
    # Pierwszy wiersz i ostatnia kolumna po maskowaniu sa puste, wiec je
    # odcinamy - pusty pas na brzegu rysunku to miejsce stracone, nie informacja.
    corr = df[cols].corr().iloc[1:, :-1]
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    texts = np.vectorize(_pl)(corr.values)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=mask, annot=texts, fmt='', cmap=style.DIVERGING_CMAP,
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                linecolor=style.BACKGROUND, annot_kws={'size': 7}, ax=ax,
                cbar_kws={'label': 'Współczynnik korelacji'})
    ax.set_title('Macierz korelacji wskaźników finansowych')
    ax.grid(visible=False)
    plt.tight_layout()
    _finish(name)


def plot_confusion(y_true, y_pred, threshold: float,
                   name: str | None = '03_macierz_pomylek',
                   title: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    texts = np.vectorize(_thousands)(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.heatmap(cm, annot=texts, fmt='', cmap=style.SEQUENTIAL_CMAP,
                xticklabels=['Zdrowa (0)', 'Upadłość (1)'],
                yticklabels=['Zdrowa (0)', 'Upadłość (1)'],
                # Linie w kolorze tla znikaja na jasnych komorkach - przy
                # silnie niezbalansowanej macierzy to wiekszosc pola.
                linewidths=0.8, linecolor=style.GRID, cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel('Rzeczywistość')
    ax.set_xlabel('Predykcja')
    ax.set_title(title or
                 f'Macierz pomyłek (próg {_pl(threshold, 3)}, zbiór testowy)')
    ax.grid(visible=False)
    plt.tight_layout()
    _finish(name)


def plot_pr_curve(y_true, y_score, pr_auc: float | None = None,
                  name: str | None = '04_krzywa_precision_recall',
                  title: str | None = None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=style.STANDARD)
    label = (f'Pole pod krzywą = {_pl(pr_auc, 4)}' if pr_auc is not None
             else 'Krzywa precyzja-czułość')
    ax.plot(recall, precision, color=style.SERIES_1, label=label)
    ax.set_xlabel('Czułość')
    ax.set_ylabel('Precyzja')
    ax.set_title(title or 'Krzywa precyzja-czułość (zbiór testowy)')
    style.decimal_comma(ax, 'xy', 1)
    ax.legend(loc='upper right')
    plt.tight_layout()
    _finish(name)


def plot_learning_curve(results: dict, metric_key: str = 'aucpr',
                        name: str | None = '05_krzywa_uczenia'):
    if not results:
        return
    key = metric_key if metric_key in results['validation_0'] else list(results['validation_0'])[0]
    epochs = range(len(results['validation_0'][key]))
    fig, ax = plt.subplots(figsize=style.STANDARD)
    # Walidacja jest tym, co sie obserwuje, wiec nosi barwe uwagi.
    ax.plot(epochs, results['validation_0'][key], color=style.REFERENCE,
            label='Trening')
    if 'validation_1' in results:
        ax.plot(epochs, results['validation_1'][key], color=style.ACCENT,
                label='Walidacja')
    ax.set_title('Krzywa uczenia modelu XGBoost')
    ax.set_xlabel('Liczba drzew')
    ax.set_ylabel('Pole pod krzywą precyzja-czułość')
    style.decimal_comma(ax, 'y', 3)
    ax.legend()
    plt.tight_layout()
    _finish(name)


def plot_failure_rate_by_size(df: pd.DataFrame, target: str = 'will_fail_next_year',
                              size_col: str = 'TotalAssets',
                              name: str | None = '09_stopa_upadlosci_wg_wielkosci_i_typu'):
    d = df.copy()
    d['size_q'] = pd.qcut(d[size_col].rank(method='first'), 4,
                          labels=['Q1 (małe)', 'Q2', 'Q3', 'Q4 (duże)'])
    rates = (d.groupby(['size_q', 'family'], observed=True)[target]
             .mean().mul(100).reset_index())
    rates['Typ'] = rates['family'].map({0.0: 'Nierodzinne', 1.0: 'Rodzinne'})
    fig, ax = plt.subplots(figsize=style.STANDARD)
    # Firmy rodzinne sa przedmiotem badania, wiec nosza barwe uwagi.
    sns.barplot(data=rates, x='size_q', y=target, hue='Typ', ax=ax,
                palette={'Nierodzinne': style.REFERENCE,
                         'Rodzinne': style.ACCENT})
    style.value_labels(ax, places=3, suffix='%')
    ax.margins(y=0.18)
    ax.set_title('Stopa upadłości według wielkości firmy')
    ax.set_xlabel('Kwartyl wielkości (aktywa ogółem)')
    ax.set_ylabel('Stopa upadłości (%)')
    style.decimal_comma(ax, 'y', 3)
    ax.legend(title='Typ firmy')
    plt.tight_layout()
    _finish(name)


def plot_optuna_history(study, name: str | None = '13_przebieg_optymalizacji'):
    """Wartosc funkcji celu w kolejnych probach oraz najlepszy wynik skumulowany.

    Pokazuje, w ktorym momencie przeszukiwania nastepowala poprawa i jak duzy
    jest rozrzut miedzy probami. Pojedyncze proby sa masa bez tozsamosci, wiec
    nosza barwe wyciszona; wyrozniony jest tylko przebieg najlepszego wyniku.
    """
    import optuna

    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    running_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=style.WIDE)
    ax.scatter(numbers, values, s=18, alpha=0.5, color=style.MUTED,
               label='Pojedyncze próby')
    ax.plot(numbers, running_best, color=style.ACCENT,
            label='Najlepszy wynik (skumulowany)')
    ax.set_title('Przebieg optymalizacji hiperparametrów')
    ax.set_xlabel('Numer próby')
    ax.set_ylabel('Pole pod krzywą precyzja-czułość (walidacja)')
    style.decimal_comma(ax, 'y', 3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    _finish(name)


def plot_param_importances(study, name: str | None = '14_waznosc_hiperparametrow',
                           seed: int = 42):
    """Wzgledna waznosc hiperparametrow dla wartosci funkcji celu (fANOVA).

    Estymator fANOVA (Hutter i in., 2014) opiera sie na lesie losowym, wiec jest
    stochastyczny - bez ustalonego ziarna kolejne wywolania daja nieco rozne
    wartosci, a parametry o zblizonej waznosci potrafia zamienic sie miejscami.
    Ziarno ustalono, by wykres byl odtwarzalny.

    Slupki nosza jedna barwe: ich dlugosc juz koduje wielkosc, wiec barwienie
    wedlug wartosci powtarzaloby te sama informacje drugim kanalem.
    """
    import optuna

    importances = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed))
    names = list(importances.keys())[::-1]
    values = [importances[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(names, values, color=style.SERIES_1)
    for y, value in enumerate(values):
        ax.text(value + max(values) * 0.012, y, _pl(value, 3), va='center',
                fontsize=9, color=style.TEXT_SECONDARY)
    ax.set_title('Względna ważność hiperparametrów')
    ax.set_xlabel('Względna ważność')
    ax.set_xlim(0, max(values) * 1.14)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    style.decimal_comma(ax, 'x', 1)
    plt.tight_layout()
    _finish(name)


def plot_risk_by_family(scores_df: pd.DataFrame,
                        score_col: str = 'Ryzyko_Upadlosci_Score',
                        name: str | None = '08_rozklad_ryzyka_rodzinne_vs_nierodzinne'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='family', y=score_col, hue='family', data=scores_df, ax=ax,
                palette={0.0: style.REFERENCE, 1.0: style.ACCENT}, legend=False,
                linecolor=style.TEXT_SECONDARY, linewidth=1.1,
                flierprops={'markerfacecolor': style.MUTED,
                            'markeredgecolor': 'none', 'markersize': 3})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Nierodzinne', 'Rodzinne'])
    ax.set_title('Rozkład oceny ryzyka według typu firmy')
    ax.set_xlabel('Typ firmy')
    ax.set_ylabel('Ocena ryzyka (XGBoost, nieskalibrowana)')
    style.decimal_comma(ax, 'y', 2)
    plt.tight_layout()
    _finish(name)


# ==========================================================================
# Wykresy bibliotek zewnetrznych (SHAP)
# ==========================================================================
# Biblioteka shap rysuje wlasne figury i nie wie nic o systemie projektu.
# Ponizsze funkcje sa jedynym dopuszczonym sposobem jej wywolania: przykrywaja
# palete, podaja mape barw jawnie (podstawienie pod shap.plots.colors nie
# zadziala - patrz plot_style.apply_shap_style) i dopisuja polski tytul.

def plot_shap_bar(shap_values, X, feature_names, max_display: int = 20,
                  name: str | None = '10_shap_sredni_wplyw_cech'):
    """Sredni bezwzgledny wplyw zmiennych na wynik modelu."""
    import shap

    style.apply_shap_style()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=list(feature_names),
                      plot_type='bar', max_display=max_display, show=False,
                      color=style.SERIES_1)
    plt.title('Średni wpływ zmiennych na wynik modelu')
    plt.tight_layout()
    _finish(name)


def plot_shap_beeswarm(shap_values, X, feature_names, max_display: int = 20,
                       name: str | None = '11_shap_rozklad_wplywu'):
    """Rozklad wplywu zmiennych - jeden punkt na obserwacje."""
    import shap

    style.apply_shap_style()
    plt.figure(figsize=(10, 8))
    # cmap i opis paska podajemy jawnie: summary_plot wiaze obie wartosci
    # domyslne przy imporcie modulu, wiec podmiana globalna ich nie dosiega.
    shap.summary_plot(shap_values, X, feature_names=list(feature_names),
                      max_display=max_display, show=False,
                      cmap=style.DIVERGING_CMAP,
                      color_bar_label='Wartość zmiennej')
    plt.title('Rozkład wpływu zmiennych na ryzyko upadłości')
    plt.tight_layout()
    _finish(name)


def plot_shap_waterfall(shap_values, explainer, X, idx: int, feature_names,
                        max_display: int = 15,
                        name: str | None = '12_shap_waterfall_pojedyncza_firma'):
    """Rozklad wyniku modelu dla pojedynczej obserwacji."""
    import shap

    style.apply_shap_style()
    explanation = shap.Explanation(values=shap_values.values[idx],
                                   base_values=explainer.expected_value,
                                   data=X[idx], feature_names=list(feature_names))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    # "N other features" shap sklada f-stringiem w kodzie, wiec slownik labels
    # go nie dosiega - podmieniamy etykiete po narysowaniu.
    ax = plt.gca()
    ax.set_yticklabels([
        re.sub(r'^(\d+) other features$', r'pozostałe zmienne (\1)', t.get_text())
        for t in ax.get_yticklabels()])
    plt.title('Rozkład wyniku modelu dla pojedynczej firmy')
    plt.tight_layout()
    _finish(name)
