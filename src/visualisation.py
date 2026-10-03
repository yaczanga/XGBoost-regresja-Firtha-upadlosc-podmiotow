"""
Funkcje wizualizacyjne dla notebooka (wykresy EDA, oceny modelu i inferencji).
Kazda funkcja tworzy i pokazuje figure; zwraca obiekt Axes/Figure gdy przydatny.

Barwy. Zaden wykres nie dobiera koloru samodzielnie - wszystkie siegaja po
tokeny z modulu styl_wykresow, opisujace ROLE koloru w danych. Zasada
generujaca systemu brzmi: cynober to przedmiot uwagi, blekit to odniesienie.
Wywolanie styl.zastosuj_styl() ustawia typografie, siatke i ramki.

Jezyk. Tekst widoczny na rysunku jest po polsku, z diakrytykami i przecinkiem
dziesietnym. Nazwy metryk takze tlumaczymy (precyzja, czulosc, pole pod krzywa
precyzja-czulosc). Komentarze i nazwy w kodzie pozostaja bez ogonkow.

Zapis do plikow. Domyslnie wykresy tylko sie wyswietlaja. Po wywolaniu
set_figures_dir(sciezka) kazdy wykres jest dodatkowo zapisywany do pliku PNG
pod nazwa podana w argumencie 'nazwa'. Sluzy to do odtworzenia zawartosci
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

import styl_wykresow as styl

# Styl ustawiamy przy imporcie. Modul istnieje wylacznie po to, by rysowac
# wykresy tego projektu, wiec nie ma stanu, w ktorym mialby rysowac w stylu
# domyslnym matplotliba - a jawne wywolanie w notebooku bywaloby pomijane.
styl.zastosuj_styl()

FIGURES_DIR: Path | None = None


def _lb(wartosc: float, miejsca: int = 2) -> str:
    """Liczba w zapisie polskim - przecinek dziesietny."""
    return f'{wartosc:.{miejsca}f}'.replace('.', ',')


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
        plt.savefig(FIGURES_DIR / f'{nazwa}.png', dpi=styl.DPI_ZAPIS,
                    bbox_inches='tight')
    plt.show()


def save_current(nazwa: str) -> None:
    """Zapisuje i wyswietla figure utworzona poza tym modulem - np. wykres SHAP,
    rysowany bezposrednio w notebooku przez biblioteke shap."""
    _finish(nazwa)


def plot_target_distribution(df: pd.DataFrame, target: str = 'will_fail_next_year',
                             nazwa: str | None = '01_rozklad_zmiennej_celu'):
    dist = df[target].value_counts()
    fig, ax = plt.subplots(figsize=styl.MALY)
    ax.bar(['Firmy zdrowe (0)', 'Upadłość (1)'],
           [dist.get(0.0, 0), dist.get(1.0, 0)],
           color=[styl.ODNIESIENIE, styl.RYZYKO], width=0.55)
    ax.set_yscale('log')
    ax.set_ylabel('Liczba obserwacji (skala logarytmiczna)')
    ax.set_title('Rozkład zmiennej celu')
    for pojemnik in ax.containers:
        podpisy = [f'{int(w):,}'.replace(',', ' ') for w in pojemnik.datavalues]
        ax.bar_label(pojemnik, labels=podpisy, fontsize=9, padding=3,
                     color=styl.TEKST_POMOCNICZY)
    plt.tight_layout()
    _finish(nazwa)


def plot_corr_matrix(df: pd.DataFrame, cols: list[str],
                     nazwa: str | None = '02_macierz_korelacji_wskaznikow'):
    # Przekatna zawsze wynosi 1,00, a oba trojkaty niosa te sama informacje.
    # Bez maski najmocniejsza barwa systemu trafia na komorki bez tresci.
    # Pierwszy wiersz i ostatnia kolumna po maskowaniu sa puste, wiec je
    # odcinamy - pusty pas na brzegu rysunku to miejsce stracone, nie informacja.
    corr = df[cols].corr().iloc[1:, :-1]
    maska = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    podpisy = np.vectorize(lambda v: _lb(v))(corr.values)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=maska, annot=podpisy, fmt='', cmap=styl.MAPA_DYWERGENCYJNA,
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                linecolor=styl.TLO, annot_kws={'size': 7}, ax=ax,
                cbar_kws={'label': 'Współczynnik korelacji'})
    ax.set_title('Macierz korelacji wskaźników finansowych')
    ax.grid(visible=False)
    plt.tight_layout()
    _finish(nazwa)


def plot_confusion(y_true, y_pred, threshold: float,
                   nazwa: str | None = '03_macierz_pomylek',
                   tytul: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    podpisy = np.vectorize(lambda v: f'{v:,}'.replace(',', ' '))(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.heatmap(cm, annot=podpisy, fmt='', cmap=styl.MAPA_SEKWENCYJNA,
                xticklabels=['Zdrowa (0)', 'Upadłość (1)'],
                yticklabels=['Zdrowa (0)', 'Upadłość (1)'],
                # Linie w kolorze tla znikaja na jasnych komorkach - przy
                # silnie niezbalansowanej macierzy to wiekszosc pola.
                linewidths=0.8, linecolor=styl.SIATKA, cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel('Rzeczywistość')
    ax.set_xlabel('Predykcja')
    ax.set_title(tytul or
                 f'Macierz pomyłek (próg {_lb(threshold, 3)}, zbiór testowy)')
    ax.grid(visible=False)
    plt.tight_layout()
    _finish(nazwa)


def plot_pr_curve(y_true, y_score, pr_auc: float | None = None,
                  nazwa: str | None = '04_krzywa_precision_recall',
                  tytul: str | None = None):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=styl.STANDARD)
    etykieta = (f'Pole pod krzywą = {_lb(pr_auc, 4)}' if pr_auc is not None
                else 'Krzywa precyzja-czułość')
    ax.plot(rec, prec, color=styl.SERIA_1, label=etykieta)
    ax.set_xlabel('Czułość')
    ax.set_ylabel('Precyzja')
    ax.set_title(tytul or 'Krzywa precyzja-czułość (zbiór testowy)')
    styl.przecinek_dziesietny(ax, 'xy', 1)
    ax.legend(loc='upper right')
    plt.tight_layout()
    _finish(nazwa)


def plot_learning_curve(results: dict, metric_key: str = 'aucpr',
                        nazwa: str | None = '05_krzywa_uczenia'):
    if not results:
        return
    key = metric_key if metric_key in results['validation_0'] else list(results['validation_0'])[0]
    epochs = range(len(results['validation_0'][key]))
    fig, ax = plt.subplots(figsize=styl.STANDARD)
    # Walidacja jest tym, co sie obserwuje, wiec nosi barwe uwagi.
    ax.plot(epochs, results['validation_0'][key], color=styl.ODNIESIENIE,
            label='Trening')
    if 'validation_1' in results:
        ax.plot(epochs, results['validation_1'][key], color=styl.AKCENT,
                label='Walidacja')
    ax.set_title('Krzywa uczenia modelu XGBoost')
    ax.set_xlabel('Liczba drzew')
    ax.set_ylabel('Pole pod krzywą precyzja-czułość')
    styl.przecinek_dziesietny(ax, 'y', 3)
    ax.legend()
    plt.tight_layout()
    _finish(nazwa)


def plot_failure_rate_by_size(df: pd.DataFrame, target: str = 'will_fail_next_year',
                              size_col: str = 'TotalAssets',
                              nazwa: str | None = '09_stopa_upadlosci_wg_wielkosci_i_typu'):
    d = df.copy()
    d['size_q'] = pd.qcut(d[size_col].rank(method='first'), 4,
                          labels=['Q1 (małe)', 'Q2', 'Q3', 'Q4 (duże)'])
    g = (d.groupby(['size_q', 'family'], observed=True)[target]
         .mean().mul(100).reset_index())
    g['Typ'] = g['family'].map({0.0: 'Nierodzinne', 1.0: 'Rodzinne'})
    fig, ax = plt.subplots(figsize=styl.STANDARD)
    # Firmy rodzinne sa przedmiotem badania, wiec nosza barwe uwagi.
    sns.barplot(data=g, x='size_q', y=target, hue='Typ', ax=ax,
                palette={'Nierodzinne': styl.ODNIESIENIE,
                         'Rodzinne': styl.AKCENT})
    styl.etykiety_wartosci(ax, miejsca=3, przyrostek='%')
    ax.margins(y=0.18)
    ax.set_title('Stopa upadłości według wielkości firmy')
    ax.set_xlabel('Kwartyl wielkości (aktywa ogółem)')
    ax.set_ylabel('Stopa upadłości (%)')
    styl.przecinek_dziesietny(ax, 'y', 3)
    ax.legend(title='Typ firmy')
    plt.tight_layout()
    _finish(nazwa)


def plot_optuna_history(study, nazwa: str | None = '13_przebieg_optymalizacji'):
    """Wartosc funkcji celu w kolejnych probach oraz najlepszy wynik skumulowany.

    Pokazuje, w ktorym momencie przeszukiwania nastepowala poprawa i jak duzy
    jest rozrzut miedzy probami. Pojedyncze proby sa masa bez tozsamosci, wiec
    nosza barwe wyciszona; wyrozniony jest tylko przebieg najlepszego wyniku.
    """
    import optuna

    proby = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    numery = [t.number for t in proby]
    wartosci = [t.value for t in proby]
    najlepszy = np.maximum.accumulate(wartosci)

    fig, ax = plt.subplots(figsize=styl.SZEROKI)
    ax.scatter(numery, wartosci, s=18, alpha=0.5, color=styl.WYCISZONA,
               label='Pojedyncze próby')
    ax.plot(numery, najlepszy, color=styl.AKCENT,
            label='Najlepszy wynik (skumulowany)')
    ax.set_title('Przebieg optymalizacji hiperparametrów')
    ax.set_xlabel('Numer próby')
    ax.set_ylabel('Pole pod krzywą precyzja-czułość (walidacja)')
    styl.przecinek_dziesietny(ax, 'y', 3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    _finish(nazwa)


def plot_param_importances(study, nazwa: str | None = '14_waznosc_hiperparametrow',
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

    waznosc = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed))
    nazwy = list(waznosc.keys())[::-1]
    wartosci = [waznosc[n] for n in nazwy]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(nazwy, wartosci, color=styl.SERIA_1)
    for y, w in enumerate(wartosci):
        ax.text(w + max(wartosci) * 0.012, y, _lb(w, 3), va='center',
                fontsize=9, color=styl.TEKST_POMOCNICZY)
    ax.set_title('Względna ważność hiperparametrów')
    ax.set_xlabel('Względna ważność')
    ax.set_xlim(0, max(wartosci) * 1.14)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    styl.przecinek_dziesietny(ax, 'x', 1)
    plt.tight_layout()
    _finish(nazwa)


def plot_risk_by_family(analiza_df: pd.DataFrame,
                        score_col: str = 'Ryzyko_Upadlosci_Score',
                        nazwa: str | None = '08_rozklad_ryzyka_rodzinne_vs_nierodzinne'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='family', y=score_col, hue='family', data=analiza_df, ax=ax,
                palette={0.0: styl.ODNIESIENIE, 1.0: styl.AKCENT}, legend=False,
                linecolor=styl.TEKST_POMOCNICZY, linewidth=1.1,
                flierprops={'markerfacecolor': styl.WYCISZONA,
                            'markeredgecolor': 'none', 'markersize': 3})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Nierodzinne', 'Rodzinne'])
    ax.set_title('Rozkład oceny ryzyka według typu firmy')
    ax.set_xlabel('Typ firmy')
    ax.set_ylabel('Ocena ryzyka (XGBoost, nieskalibrowana)')
    styl.przecinek_dziesietny(ax, 'y', 2)
    plt.tight_layout()
    _finish(nazwa)


# ==========================================================================
# Wykresy bibliotek zewnetrznych (SHAP)
# ==========================================================================
# Biblioteka shap rysuje wlasne figury i nie wie nic o systemie projektu.
# Ponizsze funkcje sa jedynym dopuszczonym sposobem jej wywolania: przykrywaja
# palete, podaja mape barw jawnie (podstawienie pod shap.plots.colors nie
# zadziala - patrz styl_wykresow.zastosuj_styl_shap) i dopisuja polski tytul.

def plot_shap_bar(shap_values, X, feature_names, max_display: int = 20,
                  nazwa: str | None = '10_shap_sredni_wplyw_cech'):
    """Sredni bezwzgledny wplyw zmiennych na wynik modelu."""
    import shap

    styl.zastosuj_styl_shap()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=list(feature_names),
                      plot_type='bar', max_display=max_display, show=False,
                      color=styl.SERIA_1)
    plt.title('Średni wpływ zmiennych na wynik modelu')
    plt.tight_layout()
    _finish(nazwa)


def plot_shap_beeswarm(shap_values, X, feature_names, max_display: int = 20,
                       nazwa: str | None = '11_shap_rozklad_wplywu'):
    """Rozklad wplywu zmiennych - jeden punkt na obserwacje."""
    import shap

    styl.zastosuj_styl_shap()
    plt.figure(figsize=(10, 8))
    # cmap i opis paska podajemy jawnie: summary_plot wiaze obie wartosci
    # domyslne przy imporcie modulu, wiec podmiana globalna ich nie dosiega.
    shap.summary_plot(shap_values, X, feature_names=list(feature_names),
                      max_display=max_display, show=False,
                      cmap=styl.MAPA_DYWERGENCYJNA,
                      color_bar_label='Wartość zmiennej')
    plt.title('Rozkład wpływu zmiennych na ryzyko upadłości')
    plt.tight_layout()
    _finish(nazwa)


def plot_shap_waterfall(shap_values, explainer, X, idx: int, feature_names,
                        max_display: int = 15,
                        nazwa: str | None = '12_shap_waterfall_pojedyncza_firma'):
    """Rozklad wyniku modelu dla pojedynczej obserwacji."""
    import shap

    styl.zastosuj_styl_shap()
    exp = shap.Explanation(values=shap_values.values[idx],
                           base_values=explainer.expected_value,
                           data=X[idx], feature_names=list(feature_names))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    # "N other features" shap sklada f-stringiem w kodzie, wiec slownik labels
    # go nie dosiega - podmieniamy etykiete po narysowaniu.
    ax = plt.gca()
    ax.set_yticklabels([
        re.sub(r'^(\d+) other features$', r'pozostałe zmienne (\1)', e.get_text())
        for e in ax.get_yticklabels()])
    plt.title('Rozkład wyniku modelu dla pojedynczej firmy')
    plt.tight_layout()
    _finish(nazwa)
