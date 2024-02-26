import pandas as pd


def save_result_df(data, output_csv):
    column_name = {
        "Dataset": [],
        "Model": [],
        "Predict Function of Model": [],
        "Metric Function": [],
        "Score": [],
    }
    for dataset, model_results in data.items():
        for model, predict_func_results in model_results.items():
            for predict_func, metric_results in predict_func_results.items():
                for metric, scores in metric_results.items():
                    average_score = sum(float(s) for s in scores) / len(scores)

                    column_name["Dataset"].append(dataset)
                    column_name["Model"].append(model)
                    column_name["Predict Function of Model"].append(predict_func)
                    column_name["Metric Function"].append(metric)
                    column_name["Score"].append(round(average_score, 8))

    result_df = pd.DataFrame(column_name)

    result_df.to_csv(output_csv, index=False)
    print(f"\n✅Evaluation results saved to {output_csv}")
    return result_df


def save_crosstab(df, output_csv, aggfunc="mean"):
    cross_tab = pd.crosstab(
        index=[df["Dataset"], df["Model"], df["Predict Function of Model"]],
        rownames=["Dataset", "Base Model", "Predict Func"],
        columns=[df["Metric Function"]],
        colnames=["Metric"],
        values=round(df["Score"] * 100, 6),
        aggfunc=aggfunc,
        margins=True,
    )

    print(cross_tab)
    cross_tab.to_csv(f"{output_csv.split('.csv')[0]}_crosstab.csv")


def add_key_if_missing(data, key1, key2, key3, key4, value=None):
    """Checks if nested keys exist, adds them if not, and sets the value."""
    if key1 not in data:
        data[key1] = {}
    if key2 not in data[key1]:
        data[key1][key2] = {}
    if key3 not in data[key1][key2]:
        data[key1][key2][key3] = {}
    if key4 not in data[key1][key2][key3]:
        data[key1][key2][key3][key4] = [value]
    else:
        data[key1][key2][key3][key4].append(value)
