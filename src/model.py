import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int, storage: str, study_name: str) -> xgb.XGBClassifier:
    
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_weight = neg / pos if pos > 0 else 1

    fixed = dict(
        objective='binary:logistic',
        eval_metric='aucpr',
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        scale_pos_weight=scale_weight,
        n_estimators = 3000,
        early_stopping_rounds = 50,
    )

    def objective(trial):
        params = dict(
            fixed,
            learning_rate   = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth       = trial.suggest_int('max_depth', 2, 5),
            min_child_weight= trial.suggest_int('min_child_weight', 10, 100),
            subsample       = trial.suggest_float('subsample', 0.4, 0.8),
            colsample_bytree= trial.suggest_float('colsample_bytree', 0.4, 0.8),
            gamma           = trial.suggest_float('gamma', 0.5, 10.0),
            reg_alpha       = trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),
            reg_lambda      = trial.suggest_float('reg_lambda', 1.0, 100.0, log=True),
            max_delta_step  = trial.suggest_int('max_delta_step', 0, 10),
        )
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose=False
        )

        y_pred = model.predict_proba(X_val)[:,1]
        return average_precision_score(y_val, y_pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    print(f"Prob zakonczonych w bazie: {done}. Do wykonania teraz: {remaining}.")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    else:
        print("Target trials reached - zwieksz n_trials w configu, by dostroic wiecej.")

    print(f"\nNajlepszy PR-AUC (walidacja): {study.best_value:.4f}")
    print(f"Najlepsze parametry: {study.best_params}")

    # Finalny model: eval na train+val -> evals_result() dostepny do krzywej uczenia
    model = xgb.XGBClassifier(**dict(fixed, **study.best_params))
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    return model, study