"""Training + k-fold evaluation pipeline.

Each (PCC, BR) × base-estimator model is trained per fold (with per-fold
StandardScaler to avoid leakage), every inference rule is run, and the
configured metrics are computed on each rule's output.
"""
import os
import time
from uuid import uuid4

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.config import BASE_DIR, SEED, KFOLD_SPLIT_NUMBER
from dacaf_mlc.datasets import read_datasets_from_folder
from dacaf_mlc.metrics_registry import PREDICT_FUNCTIONS
from dacaf_mlc.probability_classifier_chains import (
    ProbabilisticClassifierChainCustom,
    BinaryRelevance,
)
from dacaf_mlc.utils import add_key_if_missing, save_crosstab, save_result_df

ESTIMATOR_FACTORIES = {
    "lr":       lambda seed: LogisticRegression(random_state=seed, max_iter=5_000_000),
    "rf":       lambda seed: RandomForestClassifier(random_state=seed, n_estimators=100, n_jobs=1),
    "adaboost": lambda seed: AdaBoostClassifier(random_state=seed, n_estimators=50),
}


def model_display_key(model):
    """Disambiguate (PCC vs BR) × base estimator. Used as the row key in result CSVs."""
    short = {
        "ProbabilisticClassifierChainCustom": "PCC",
        "BinaryRelevance": "BR",
    }.get(type(model).__name__, type(model).__name__)
    return f"{short}_{model.base_estimator.__class__.__name__}"


def calculate_metrics(Y_true, Y_pred_or_scores, metric_funcs):
    """Calculate metrics. Y_pred_or_scores is binary for standard metrics, continuous for ranking."""
    scores = []
    for metric in metric_funcs:
        name = metric["name"]
        func = metric["func"]
        opts = metric.get("options", {})
        try:
            value = func(Y_true, Y_pred_or_scores, **opts) if opts else func(Y_true, Y_pred_or_scores)
            scores.append({"Metric Name": name, "Metric Function": func.__name__, "Score": f"{value:.5f}"})
        except Exception as e:
            print(f"Error calculating {name}: {e}")
    return scores


def training_model(model, X_train, Y_train, cache_key=None):
    """Train the model and assign a cache key for prediction reuse."""
    if cache_key is None:
        cache_key = uuid4().hex
    try:
        start = time.time()
        print(f"Training {model_display_key(model)} ...")
        model.set_cache_key(cache_key)
        model.fit(X_train, Y_train)
        print(f"Training time: {time.time() - start:.3f}s")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise


def evaluate_model(model, X_test, Y_test, predict_funcs):
    """Run all predict_* functions; each pf carries its own metric list."""
    results = []
    for pf in predict_funcs:
        start = time.time()
        Y_pred_or_scores = getattr(model, pf["func"])(X_test)
        print(f"Predict time: {time.time() - start:.3f}s [{pf['name']}]")
        scores = calculate_metrics(Y_test, Y_pred_or_scores, pf["metrics"])
        results.append({"predict_name": pf["name"], "score_metrics": scores})
    return results


def prepare_model_to_evaluate(estimator_names=None, seed=SEED):
    """Return the list of (PCC, BR) × (base_estimator) models to evaluate.

    estimator_names: iterable of keys from ESTIMATOR_FACTORIES, or None for all.
    """
    if estimator_names is None:
        estimator_names = list(ESTIMATOR_FACTORIES)
    models = []
    for name in estimator_names:
        est = ESTIMATOR_FACTORIES[name](seed)
        models.append(ProbabilisticClassifierChainCustom(est))
        models.append(BinaryRelevance(est))
    return models


def evaluate_kfold(
    dataset_handler,
    evaluated_models,
    predict_functions,
    train_index,
    test_index,
    kfold_index,
):
    """Evaluate all models on one k-fold split with proper per-fold scaling."""
    print(f"\nFold {kfold_index} ...")
    dataset_name = dataset_handler.dataset_name

    X_train_raw = dataset_handler.X[train_index]
    X_test_raw = dataset_handler.X[test_index]
    y_train = dataset_handler.Y[train_index]
    y_test = dataset_handler.Y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    fold_results = {dataset_name: {}}
    for model in evaluated_models:
        m_key = model_display_key(model)
        fold_results[dataset_name][m_key] = {}

        model = training_model(
            model, X_train, y_train,
            cache_key=f"{dataset_name}_{m_key}_fold{kfold_index}",
        )
        results = evaluate_model(model, X_test, y_test, predict_functions)

        for result in results:
            pf_name = result["predict_name"]
            fold_results[dataset_name][m_key][pf_name] = {}
            for sm in result["score_metrics"]:
                fold_results[dataset_name][m_key][pf_name][sm["Metric Name"]] = sm["Score"]

    return fold_results


def _n_parallel_folds():
    """Honour Slurm's CPU allocation when present; else use all local cores."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm_cpus) if slurm_cpus else -1


def run_single(dataset_name, seed, estimator_names, output_dir):
    """Run k-fold CV for one (dataset, seed, estimator-set) combo and write one CSV.

    Output: <output_dir>/<dataset>/seed<seed>_<est-tag>.csv (+ _crosstab.csv).
    `est-tag` is "all" if every estimator is requested, otherwise a `-` join.
    """
    folder_path = os.path.join(BASE_DIR, "datasets")
    out_subdir = os.path.join(output_dir, dataset_name)
    os.makedirs(out_subdir, exist_ok=True)

    est_tag = "all" if set(estimator_names) == set(ESTIMATOR_FACTORIES) else "-".join(estimator_names)
    output_csv = os.path.join(out_subdir, f"seed{seed}_{est_tag}.csv")

    evaluated_models = prepare_model_to_evaluate(estimator_names=estimator_names, seed=seed)

    (dataset_handler,) = list(read_datasets_from_folder(folder_path, [dataset_name]))
    print(f"\nEvaluating: {dataset_name} (seed={seed}, estimators={estimator_names})")
    t0 = time.time()

    job_results = Parallel(n_jobs=_n_parallel_folds())(
        delayed(evaluate_kfold)(
            dataset_handler,
            evaluated_models,
            PREDICT_FUNCTIONS,
            train_index,
            test_index,
            kfold_index,
        )
        for kfold_index, (train_index, test_index) in enumerate(
            dataset_handler.get_cross_validation_folds(
                n_splits=KFOLD_SPLIT_NUMBER, random_state=seed
            )
        )
    )

    dataset_results = {dataset_name: {}}
    for fold_result in job_results:
        if fold_result is None:
            continue
        for model in evaluated_models:
            m_key = model_display_key(model)
            for pf in PREDICT_FUNCTIONS:
                for mf in pf["metrics"]:
                    add_key_if_missing(
                        dataset_results, dataset_name, m_key, pf["name"], mf["name"],
                        fold_result[dataset_name][m_key][pf["name"]][mf["name"]],
                    )

    result_df = save_result_df(dataset_results, output_csv)
    save_crosstab(result_df, output_csv)
    print(f"Dataset evaluation time: {time.time() - t0:.1f}s")
    return output_csv
