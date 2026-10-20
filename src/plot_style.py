"""
System wizualny projektu - tokeny barw, typografia i ustawienia matplotlib.

Funkcje rysujace nie dobieraja koloru samodzielnie, tylko siegaja po token
opisujacy ROLE koloru w danych; zasada przydzialu brzmi "cynober to przedmiot
uwagi, blekit to odniesienie". Para blekit-cynober osiaga delta E 15,6 przy
protanopii wobec progu 8 i ponad 3:1 kontrastu wobec bieli.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Rola: tozsamosc serii
# System niesie tozsamosc DWIEMA barwami. Trzecia kategoria idzie w ORDINAL_RAMP
# (gdy jest uporzadkowana), w ACCENT na tle MUTED albo na osobny panel.
BLUE = '#006c98'
VERMILION = '#c64e31'

SERIES_1 = BLUE          # odniesienie, norma, grupa kontrolna
SERIES_2 = VERMILION     # przedmiot uwagi, kategoria badana

# Rola: stan (wartosciowanie)
RISK = VERMILION         # upadlosc, blad, zdarzenie negatywne
REFERENCE = BLUE         # firma zdrowa, trafienie, stan bazowy

# Rola: akcent (jeden wyrozniony element na tle masy)
ACCENT = VERMILION
MUTED = '#7b8f9e'        # masa elementow BEZ tozsamosci; nigdy nie trafia
                         # do legendy jako nazwana kategoria

# Rola: neutralna (tusz nie-danych)
TEXT = '#22292e'            # tytul, etykiety osi             14,74:1
TEXT_SECONDARY = '#5c6b78'  # etykiety wartosci, znaczniki     5,48:1
AXES = '#90a0ad'
GRID = '#dde2e6'
BACKGROUND = '#ffffff'

# Rola: porzadkowa (kategorie uporzadkowane: kwartyle, przedzialy, etapy)
# Jeden odcien, monotoniczna jasnosc. Jasny koniec ma kontrast 2,21:1 wobec
# bieli, wiec pierwszy stopien nie znika na tle.
ORDINAL_RAMP = ['#8fb4c7', '#6b9db5', '#3a7694', '#1a5572', '#06324a']

# Rola: sekwencyjna (wielkosc: macierz pomylek, waznosc cech)
# Jasny koniec siega bieli celowo: wartosc zerowa ma czytac sie jako puste tlo,
# a nie jako najnizszy stopien skali. To rampa ciagla, nie porzadkowa.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'project_sequential',
    ['#ffffff', '#c3d6e0', '#7fa8bd', '#3a7694', '#06324a'])

# Rola: dywergencyjna (biegunowosc wokol zera, np. macierz korelacji)
# Blekit = wartosci ujemne, cynober = dodatnie, szary neutralny w zerze.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'project_diverging',
    ['#006c98', '#b6cad6', '#f2f2f2', '#e8cdc3', '#c64e31'])

# Proporcje figur - zamknieta lista
SMALL = (6.5, 4.0)       # jedna seria, prosty komunikat
STANDARD = (9.0, 5.5)    # domyslny
WIDE = (12.0, 6.0)       # wiele kategorii na osi poziomej, szeregi
SQUARE = (7.5, 7.0)      # macierze

SAVE_DPI = 150


def apply_style() -> None:
    """Ustawia rcParams. Wywolac raz, przed pierwszym rysunkiem."""
    mpl.rcParams.update({
        'figure.figsize': STANDARD,
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'savefig.facecolor': BACKGROUND,
        'savefig.dpi': SAVE_DPI,
        'savefig.bbox': 'tight',

        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titlelocation': 'left',
        'axes.titlepad': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        'text.color': TEXT,
        'axes.titlecolor': TEXT,
        'axes.labelcolor': TEXT_SECONDARY,
        'xtick.color': TEXT_SECONDARY,
        'ytick.color': TEXT_SECONDARY,

        # Ramke nosi tylko os odnoszaca - gorna i prawa nie niosa informacji.
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.edgecolor': AXES,
        'xtick.major.size': 0,
        'ytick.major.size': 0,

        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,

        'legend.frameon': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.prop_cycle': mpl.cycler(color=[BLUE, VERMILION]),
    })




def decimal_comma(ax, axis: str = 'y', places: int = 2) -> None:
    """Zamienia kropke dziesietna na przecinek na podanych osiach.

    Etykiety sa po polsku, wiec separatorem dziesietnym jest przecinek.
    """
    formatter = FuncFormatter(
        lambda value, _: f'{value:.{places}f}'.replace('.', ','))
    if 'x' in axis:
        ax.xaxis.set_major_formatter(formatter)
    if 'y' in axis:
        ax.yaxis.set_major_formatter(formatter)


def value_labels(ax, places: int = 3, suffix: str = '', **kwargs) -> None:
    """Podpisuje slupki wartoscia, z przecinkiem dziesietnym.

    Etykieta nosi barwe tekstu, nie barwe serii - liczba ma uzupelniac dane,
    nie konkurowac z nimi o uwage. Separator dziesietny musi byc ten sam co na
    osiach, inaczej jeden rysunek miesza dwie konwencje zapisu liczby.

    Etykiety stosujemy wybiorczo - tam, gdzie liczba jest przedmiotem odczytu.
    Podpisanie kazdego punktu zamienia wykres w tabele i zasloni dane, gdy
    slupki sasiednich serii sa waskie.
    """
    for container in ax.containers:
        texts = [f'{value:.{places}f}'.replace('.', ',') + suffix
                 for value in container.datavalues]
        ax.bar_label(container, labels=texts, fontsize=9, padding=2,
                     color=TEXT_SECONDARY, **kwargs)
