# Możliwości i ograniczenia uczenia maszynowego w predykcji upadłości przedsiębiorstw na przykładzie firm rodzinnych

Kod i wyniki pracy magisterskiej napisanej na Wydziale Nauk Ekonomicznych Uniwersytetu
Warszawskiego (kierunek: Informatyka i Ekonometria, sierpień 2026).

Projekt odpowiada na dwa pytania. Po pierwsze — czy na podstawie wskaźników finansowych da się
realnie wskazać przedsiębiorstwa zagrożone upadłością, a jeśli model cokolwiek przewiduje, to
skąd bierze się jego moc predykcyjna. Po drugie — czy spółki rodzinne upadają rzadziej niż
nierodzinne i czy różnica ta przetrwa kontrolę wielkości oraz zmianę metody estymacji.

Badanie celowo rozdziela te cele na **model predykcyjny** (XGBoost) i **model inferencyjny**
(logit hazardu, regresja Firtha, test permutacyjny). Dobry klasyfikator nie jest dowodem na
związek przyczynowy, a model inferencyjny nie służy do rankingowania firm.

## Hipotezy

| | Treść | Wynik |
|---|---|---|
| **H1** | Zdolność predykcyjna modelu upadłości wynika głównie z cech administracyjnych zbioru danych, a nie z ekonomicznej treści wskaźników finansowych. | potwierdzona |
| **H2** | Przedsiębiorstwa rodzinne charakteryzują się niższym ryzykiem upadłości niż nierodzinne. | potwierdzona |

## Dane

Źródłem jest baza **Orbis** (Bureau van Dijk) — sprawozdania finansowe polskich spółek za lata
**2013–2020**. Jednostką obserwacji jest **firmo-rok**, co odpowiada modelowi hazardu w czasie
dyskretnym (Allison, 1982) i pozwala uniknąć obciążeń właściwych klasycznym modelom opartym na
jednej obserwacji na firmę (Shumway, 2001) oraz parowaniu firm upadłych ze zdrowymi
(Zmijewski, 1984).

| | Wartość |
|---|---|
| Panel wyjściowy | 2 723 508 firmo-lat |
| Po filtrze braków (ponad 12 z 17 wskaźników) | 1 317 608 firmo-lat, 262 609 podmiotów |
| Zdarzenia (upadłości) | 372 |
| Częstość zdarzenia | 1 na 3 541 firmo-lat (0,028%) |

Filtr usunął 51,6% panelu, tracąc przy tym zaledwie **dwie** obserwacje pozytywne. To pierwsza
przesłanka, że braki danych nie są losowe i same w sobie niosą informację o statusie podmiotu.

> **Dane nie są częścią repozytorium.** Plik `dane.dta` waży 3,48 GB (limit pojedynczego pliku na
> GitHubie to 100 MB), a zbiór Orbis jest licencjonowany i nie podlega redystrybucji. Aby
> odtworzyć badanie, trzeba dysponować własnym dostępem do bazy i umieścić wyciąg
> w `data/raw/dane.dta`.

## Metody

**Konstrukcja zmiennej celu.** Zmienna przyjmuje 1, jeżeli *następny* rok jest rokiem formalnej
upadłości. Usunięto obserwacje z roku zdarzenia i lat późniejszych, ponieważ od ogłoszenia
upadłości podmiot nie jest już narażony na ryzyko. Zastosowano też cenzurowanie prawostronne —
ostatni obserwowany rok firm bez zdarzenia odpada, bo nie znamy stanu w roku następnym. Bez tego
przesunięcia model nie przewidywałby przyszłości, tylko rozpoznawał stan już trwający, co dałoby
pozornie świetne wyniki.

**Likwidacje nie są zdarzeniem**, ale podmioty zlikwidowane pozostają w zbiorze jako nieupadłe.
Ich usunięcie groziłoby błędem doboru próby, bo likwidacja jest silnie powiązana ze statusem
rodzinnym.

**Przygotowanie danych.** Winsoryzacja dwustronna na poziomie 1%, z progami liczonymi **wyłącznie
na zbiorze uczącym** — inaczej informacja z walidacji i testu przeciekłaby do procesu
przygotowania danych. Braki pozostawiono bez imputacji, bo XGBoost uczy kierunku podziału dla
braku osobno w każdym węźle, a sam fakt braku niesie tu informację. Firmy o zerowych lub ujemnych
aktywach zachowano z flagą zamiast usuwać: to 5,5% panelu, ale aż 39% zdarzeń.

**Podział próby na poziomie firm** (68/12/20), nie wierszy. Przy podziale po wierszach różne lata
tej samej spółki trafiłyby jednocześnie do uczenia i testu, a model rozpoznawałby konkretne
podmioty zamiast ogólnych prawidłowości.

| Podzbiór | Firmo-lata | Upadłości |
|---|---|---|
| Uczący | 896 718 | 256 |
| Walidacyjny | 157 197 | 46 |
| Testowy | 263 693 | 70 |

**Model predykcyjny.** XGBoost z ważeniem klasy mniejszościowej (`scale_pos_weight`), strojony
metodą optymalizacji bayesowskiej (Optuna, TPE) — **500 prób, 9 hiperparametrów**, funkcja celu
to PR-AUC na walidacji. Główną miarą oceny jest **PR-AUC, a nie ROC-AUC**: przy ponad milionie
firm zdrowych nawet kilkanaście tysięcy fałszywych alarmów stanowi ułamek procenta, więc ROC
pozostaje wysokie niezależnie od praktycznej użyteczności modelu (Saito i Rehmsmeier, 2015).

**Model inferencyjny.** Przy zaledwie dziesięciu zdarzeniach w grupie rodzinnej klasyczna metoda
największej wiarygodności jest zawodna (King i Zeng, 2001). Wnioskowanie opiera się więc na
zbieżności trzech procedur o odmiennych założeniach: logitu hazardu z błędami standardowymi
klastrowanymi po firmie, regresji Firtha z penalizacją Jeffreysa oraz testu permutacyjnego
klastrowanego po firmie. Firth i test permutacyjny są zaimplementowane od zera
w [`src/inference.py`](src/inference.py), bez zależności zewnętrznych.

## Kluczowe wyniki

### Predykcja jest słaba — i nie opiera się na finansach

Na zbiorze testowym, nieużywanym na żadnym etapie uczenia ani strojenia:

| Miara | Wartość |
|---|---|
| **PR-AUC** (oszacowanie punktowe) | **0,032** |
| PR-AUC — mediana bootstrapu klastrowanego po firmie | 0,034 [0,018; 0,062] |
| ROC-AUC | 0,980 |
| PR-AUC na walidacji (najlepsza z 500 prób) | 0,208 |
| Precyzja przy czułości 0,90 | 0,004 |
| Przewaga nad losowym uporządkowaniem | ~121× (prewalencja 0,000265) |

Rozbieżność między PR-AUC 0,032 a ROC-AUC 0,980 to podręcznikowy objaw skrajnego
niezbalansowania klas. Sam ROC-AUC prowadziłby do wniosku o modelu niemal doskonałym.
W praktyce przy progu wychwytującym 90% upadłości **na każdą trafną wskazówkę przypada około
250 fałszywych alarmów**. Uporządkowanie generowane przez model jest ok. 120 razy lepsze od
losowego, ale w kategoriach bezwzględnych nie jest to użyteczne narzędzie przesiewowe.

**Test źródła sygnału** rozstrzyga, skąd bierze się ta resztkowa moc predykcyjna:

| Zestaw cech | PR-AUC |
|---|---|
| Wszystkie cechy | 0,034 |
| Bez cech „martwej” firmy (wielkość, status sprawozdawczy) | 0,042 |
| Tylko wskaźniki finansowe | 0,019 |

Wartości w tej tabeli pochodzą z osobnych, nieustrojonych modeli o jednakowych
hiperparametrach (`signal_source_test`), więc nie są wprost porównywalne z PR-AUC 0,032 modelu
strojonego — porównywalne są wyłącznie między sobą.

Model oparty **wyłącznie na wskaźnikach finansowych wypada najgorzej**, a usunięcie miar
wielkości i statusu sprawozdawczego trafności nie pogarsza. Potwierdza to analiza wartości
Shapleya: ranking wpływu jest zdominowany przez cechy strukturalne, a najwyższe ryzyko model
przypisuje podmiotom o zerowych aktywach i niekompletnej sprawozdawczości — nie firmom
o pogarszającej się rentowności czy płynności. Model rozpoznaje „martwe” firmy, a nie narastające
trudności finansowe (H1).

Wszystkie trzy warianty mieszczą się w przedziale ufności wariantu pełnego, więc przy
siedemdziesięciu zdarzeniach w zbiorze testowym różnic między nimi nie należy nadinterpretować.

### Firmy rodzinne upadają około 11 razy rzadziej

| Typ firmy | Stopa upadłości | Upadłości | Firmo-lata |
|---|---|---|---|
| Nierodzinne | 0,0360% | 362 | 1 005 543 |
| Rodzinne | 0,0032% | 10 | 312 065 |

Gdyby spółki rodzinne upadały w tempie nierodzinnych, należałoby oczekiwać około 110 upadłości
zamiast 10. Różnica nie znika po kontrolach:

| Model | Współczynnik `family` | Błąd std. | Iloraz szans |
|---|---|---|---|
| M1 (bez kontroli) | −2,42\*\*\* | (0,32) | 0,089 |
| M2 (+ rozmiar, staż, flaga aktywów) | −2,50\*\*\* | (0,32) | 0,082 |
| M3 (+ efekty stałe sektora i roku) | −3,54\*\*\* | (0,46) | 0,029 — numerycznie niestabilny |
| **Firth (specyfikacja M2)** | **−2,45\*\*\*** | **(0,31)** | **0,086**, profilowy CI [0,044; 0,151] |

\*\*\* istotność na poziomie 0,001.

Gdyby niższa upadalność firm rodzinnych brała się wyłącznie z ich większego rozmiaru, dodanie
kontroli wielkości osłabiłoby efekt — a tego nie widać. **Test permutacyjny**: wśród 500 losowych
przetasowań statusu rodzinnego na poziomie firmy nie wystąpił ani raz efekt co najmniej tak silny
jak zaobserwowany (empiryczne *p* = 0,002).

Dominującym predyktorem w modelu inferencyjnym jest flaga zerowych aktywów — ponad 12 razy wyższe
szanse upadłości — a logarytm aktywów po jej uwzględnieniu przestaje być istotny, spójnie
z częścią predykcyjną. Efekt statusu rodzinnego pozostaje silny i stabilny nawet obok tych
dominujących kontroli (H2).

## Struktura projektu

```
magisterka/
├── data/
│   └── raw/                    dane źródłowe (poza repozytorium)
├── notebooks/
│   └── 01_upadlosc_a_status_rodzinny.ipynb    pełny przebieg badania
├── src/
│   ├── data_load.py            wczytanie panelu z pliku .dta
│   ├── pre_processing.py       target hazardu, czyszczenie, cechy, podział, winsoryzacja
│   ├── model.py                XGBoost + strojenie Optuną (pojedynczy podział)
│   ├── cross_validation.py     walidacja krzyżowa z podziałem na poziomie firm
│   ├── validation.py           próg decyzyjny, bootstrap klastrowany, test źródła sygnału
│   ├── inference.py            Firth, logit hazardu, test permutacyjny, sekcje NACE
│   └── visualisation.py        wykresy EDA i oceny modelu
├── models/
│   ├── xgboost_upadlosci.json  wytrenowany model
│   └── optuna_upadlosci.db     baza 500 prób strojenia (wznawialna)
├── reports/                    listy firm zagrożonych (poza repozytorium — NIP i nazwy)
├── requirements.txt
└── .env.example
```

Notebook `01_` prowadzi przez całe badanie: wczytanie danych, konstrukcję targetu, diagnostykę
braków, czyszczenie panelu, EDA, strojenie, ocenę, test źródła sygnału, część inferencyjną
i analizę SHAP. Logika obliczeniowa siedzi w `src/` — notebook ją wywołuje i opisuje.

## Uruchomienie

```bash
python -m venv venv
pip install -r requirements.txt
```

Środowisko aktywuje się przez `venv\Scripts\activate` na Windowsie albo
`source venv/bin/activate` na Linuksie i macOS. Następnie umieść wyciąg z bazy Orbis
w `data/raw/dane.dta` i uruchom notebook:

```bash
jupyter lab notebooks/01_upadlosc_a_status_rodzinny.ipynb
```

Notebook sam ustala korzeń projektu, więc działa niezależnie od katalogu uruchomienia.
Testowano na Pythonie 3.14.5.

**Czas obliczeń.** Strojenie to 500 prób XGBoost na panelu 1,3 mln wierszy — liczone w godzinach.
Studium Optuny zapisuje się w `models/optuna_upadlosci.db`, więc przerwany proces wznawia się od
ostatniej ukończonej próby. Dołączona baza zawiera komplet 500 prób, a zmienna `TARGET_TRIALS`
w notebooku steruje tym, ile ma ich być łącznie. Test permutacyjny (sekcja 15b) jest opcjonalny
i również długi.

## Ograniczenia

- **Status rodzinny** ujęto binarnie, zgodnie z klasyfikacją dostępną w Orbis. Miara nie
  różnicuje intensywności kontroli ani pokolenia sprawującego władzę, a literatura wskazuje,
  że przewaga firm rodzinnych zanika w spółkach kierowanych przez potomków założyciela
  (Villalonga i Amit, 2006).
- **Wnioskowanie opiera się na 10 zdarzeniach** w grupie rodzinnej. Stąd regresja Firtha,
  profilowe przedziały ufności i test permutacyjny zamiast zwykłego logitu.
- **Staż firmy** to czas obecności w bazie, a nie rzeczywisty wiek od założenia (ucięcie
  lewostronne). Hipotezy o „obciążeniu nowością” nie da się tu przetestować.
- **Likwidacje nie są zdarzeniem**, więc część różnicy między grupami może odzwierciedlać
  odmienny kanał wyjścia z rynku, a nie samo ryzyko niewypłacalności.
- **Sprawozdanie za rok poprzedzający** zdarzenie bywa publikowane już po ogłoszeniu upadłości,
  więc predykcja nie odpowiada w pełni decyzji możliwej do podjęcia z rocznym wyprzedzeniem.
- **Zwracany score nie jest skalibrowanym prawdopodobieństwem** — uniemożliwia to ważenie klasy.
  Służy porządkowaniu firm według ryzyka.
- **Model M3** (z efektami stałymi sektora i roku) jest numerycznie niestabilny i nie stanowi
  podstawy wnioskowania.

## Walidacja krzyżowa z podziałem na poziomie firm

Rozszerzenie wykraczające poza wersję pracy, zaimplementowane
w [`src/cross_validation.py`](src/cross_validation.py) i w sekcji 9b notebooka.

W podstawowym przebiegu jeden zbiór walidacyjny pełni trzy funkcje naraz — wczesne zatrzymanie,
wybór hiperparametrów i dobór progu decyzyjnego — przy zaledwie 46 zdarzeniach. Wybór najlepszego
z 500 zestawów częściowo dopasowuje się do przypadkowych właściwości tego zbioru, co widać po
rozjeździe PR-AUC: **0,208 na walidacji wobec 0,034 na teście**. Praca wskazuje walidację krzyżową
jako właściwą alternatywę (rozdz. 3.6.4), ale nie została ona w niej przeprowadzona.

Implementacja respektuje dwa wymogi tych danych:

- **podział na poziomie firm** (grupa = `TIN`), bo różne lata tej samej spółki są skorelowane;
- **stratyfikacja po zmiennej celu**, bo przy jednym zdarzeniu na 3 541 firmo-lat zwykły
  `GroupKFold` generuje foldy o skrajnie różnej liczbie zdarzeń, przez co średnia z foldów traci
  sens. Oba warunki spełnia `StratifiedGroupKFold`.

Dodatkowo `scale_pos_weight` liczony jest osobno dla każdego foldu, z jego własnej części uczącej
— inaczej do wagi klasy przeciekałaby informacja o rozkładzie zdarzeń w części walidacyjnej.
Strojenie korzysta z prunera medianowego, który przerywa próby odstające in minus po pierwszych
foldach; bez tego koszt byłby pięciokrotnie wyższy niż przy pojedynczym podziale. Model końcowy
douczany jest na całym zbiorze uczącym z liczbą drzew równą medianie z foldów.

**Przenoszenie progu decyzyjnego.** Predykcje out-of-fold pochodzą z pięciu różnych modeli,
z których każdy ma własną skalę score'ów. Próg w wartościach bezwzględnych nie przenosi się więc
wiernie na model końcowy, a PR-AUC liczone na połączonych predykcjach OOF wychodzi niżej niż
średnia z foldów — i nie jest to błąd. Zamiast progu przenoszony jest **odsetek alarmów**
potrzebny do osiągnięcia zadanej czułości; wielkość ta jest niezmiennicza względem monotonicznego
przeskalowania score'ów (`alarm_rate_from_oof` → `threshold_from_alarm_rate`).

Zbiór testowy nie uczestniczy w walidacji krzyżowej na żadnym etapie.

> **Koszt obliczeń.** Każda próba to pięć dopasowań XGBoost zamiast jednego. Studium zapisuje się
> w osobnej bazie `models/optuna_upadlosci_cv.db`, więc proces można przerwać i wznowić.
> Domyślne `N_TRIALS_CV = 100` w notebooku jest punktem wyjścia, nie wartością docelową.

### Wynik: pojedynczy podział zawyżał oszacowanie 3,6-krotnie

Przebieg na 100 próbach, 5 foldach, 1 053 915 firmo-latach i 302 zdarzeniach (zbiór uczący
i walidacyjny połączone; test nietknięty). Foldy wyszły równe — po 60–61 zdarzeń każdy, zero firm
po obu stronach podziału.

| | Oszacowanie ze strojenia | PR-AUC na teście | Rozjazd |
|---|---|---|---|
| Pojedynczy podział | 0,208 (1 zbiór, 46 zdarzeń) | 0,032 | **+0,185** |
| Walidacja krzyżowa | 0,075 ± 0,024 (5 foldów, 302 zdarzenia) | 0,023 | **+0,052** |

To jest właściwy wniosek z tego rozszerzenia. Walidacja krzyżowa **nie dała lepszego modelu** —
jego PR-AUC na teście (0,023) jest nawet nieco niższe i mieści się w przedziale ufności modelu
z pojedynczego podziału [0,013; 0,047], więc różnicy przy 70 zdarzeniach testowych nie można
rozstrzygnąć. Dała natomiast **uczciwe oszacowanie przed zobaczeniem testu**: optymizm spadł
z +0,185 do +0,052, czyli 3,6-krotnie.

Potwierdza to hipotezę z rozdz. 3.6.4 pracy: wartość 0,208 nie była miarą jakości modelu, lecz
w znacznej części efektem wyboru najlepszego z 500 zestawów na 46 zdarzeniach. PR-AUC na foldach
rozrzuciło się w zakresie 0,047–0,113, co pokazuje, jak niestabilna jest ta miara przy
kilkudziesięciu zdarzeniach — a tyle właśnie miał pojedynczy zbiór walidacyjny.

Przeniesienie progu przez odsetek alarmów zadziałało: wymagane 6,16% oznaczonych firmo-lat dało
na teście czułość 0,944 przy zadanych 0,90.

## Dane osobowe

Pełna lista firm zagrożonych (`reports/lista_zagrozonych_firm_xgboost.csv`) zawiera NIP-y i nazwy
ponad 15 tysięcy przedsiębiorstw wraz z przypisaną im oceną ryzyka upadłości. Plik jest wykluczony
z repozytorium, a notebook drukuje podgląd bez kolumn identyfikujących.

## Autor

Jan Dziuba — Wydział Nauk Ekonomicznych, Uniwersytet Warszawski.
Praca pod kierunkiem dr Anety Dzik-Walczak, Katedra Statystyki i Ekonometrii WNE UW.
