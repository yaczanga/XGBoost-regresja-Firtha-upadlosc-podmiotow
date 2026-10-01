import pandas as pd
import numpy as np

def build_hazard_target(df: pd.DataFrame, event_year_cols: list[str]) -> pd.DataFrame:
    event_year = (df.groupby('TIN')[event_year_cols].min()
              .min(axis=1).rename('event_year'))
    df = df.merge(event_year.reset_index(), on='TIN', how='left')

    # --- 2b. Target hazardu: nastepny rok = rok zdarzenia ---
    df['will_fail_next_year'] = (df['year'] + 1 == df['event_year']).astype(float)

    # --- 2c. Usuwamy wiersze w roku zdarzenia i pozniej ---
    df = df[df['event_year'].isna() | (df['year'] < df['event_year'])].copy()

    # --- 2d. Prawa cenzura: firmy BEZ zdarzenia -> usuwamy ostatni obserwowany rok ---
    last_obs = df.groupby('TIN')['year'].transform('max')
    df = df[~(df['event_year'].isna() & (df['year'] == last_obs))].copy()
    
    return df

def clean_panel(df: pd.DataFrame, ratio_cols: list[str], max_missing: int) -> pd.DataFrame:
    df['zero_assets'] = (df['TotalAssets'] <= 0).astype(int)

    df['missing_count'] = df[ratio_cols].isnull().sum(axis=1)

    df = df[df['missing_count'] <= max_missing].copy()

    return df

def add_panel_features(df: pd.DataFrame, ratio_cols: list[str]) -> pd.DataFrame:
    
    delta_cols = []
    for col in ratio_cols:
        dname = f'{col}_yoy'
        df[dname] = df.groupby('TIN')[col].diff()
        delta_cols.append(dname)
        
    df['firm_age'] = df['year'] - df.groupby('TIN')['year'].transform('min')
    df['nace_div'] = df['NACECode'] // 100
    
    return df


    
def split_firms(df: pd.DataFrame, test_pct: float, val_pct: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unique_firms = df['TIN'].unique()
    rng.shuffle(unique_firms)
    n = len(unique_firms)
    test_tins = set(unique_firms[:int(test_pct*n)])
    val_tins = set(unique_firms[int(test_pct*n):int((test_pct+val_pct)*n)])

    def assign(t):
        if t in test_tins: return 'test'
        if t in val_tins: return 'val'
        return 'train'
    df['split'] = df['TIN'].map(assign)

    return df


def build_model_features(family_cols: list[str], ratio_cols: list[str],
                         extra: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Zwraca (model_features, delta_cols).

    Cechy dynamiczne (yoy) i pochodne musza byc wczesniej dodane przez
    add_panel_features(). 'year' kalendarzowy jest CELOWO pomijany
    (zdarzenia z likwidacji datowane na ostatni rok firmy -> sztuczne skupienie).
    """
    delta_cols = [f'{c}_yoy' for c in ratio_cols]
    default_extra = ['firm_age', 'zero_assets', 'missing_count',
                     'nace_div', 'Sales']
    extra = default_extra if extra is None else extra
    model_features = family_cols + ratio_cols + delta_cols + extra
    return model_features, delta_cols


def winsorize_train_only(df: pd.DataFrame, cols: list[str], clip: float = 0.01,
                         split_col: str = 'split', train_value: str = 'train') -> pd.DataFrame:
    """Winsoryzacja dwustronna: progi liczone WYLACZNIE na treningu, stosowane wszedzie.

    Eliminuje wyciek informacji z prob walidacyjnej i testowej.
    """
    train_mask = df[split_col] == train_value
    for col in cols:
        if col not in df.columns:
            continue
        lo = df.loc[train_mask, col].quantile(clip)
        hi = df.loc[train_mask, col].quantile(1 - clip)
        df[col] = df[col].clip(lo, hi)
    return df


def make_xy(df: pd.DataFrame, model_features: list[str], split_name: str,
            id_cols_seq: list[str], target: str = 'will_fail_next_year',
            split_col: str = 'split'):
    """Macierze (X, y, groups, ids) dla danego podzbioru. Jeden wiersz = firmo-rok."""
    sub = df[df[split_col] == split_name]
    X = sub[model_features].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    groups = sub['TIN'].to_numpy()
    ids = sub[[c for c in id_cols_seq if c in sub.columns]].reset_index(drop=True)
    return X, y, groups, ids

