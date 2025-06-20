import argparse
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Run LightAutoML model training')
    parser.add_argument('--df_name', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--task_type', type=str, required=True, help='Type of task (e.g., "binary", "multiclass", "regression")')
    parser.add_argument('--task_metric', type=str, required=True, help='metric for the task (e.g., "auc", "f1", "rmse")')
    parser.add_argument('--target', type=str, required=True, help='Name of the target column in the dataset')
    args = parser.parse_args()

    df_name = args.df_name 
    task_type = args.task_type
    task_metric = args.task_metric
    target = args.target

    df = pd.read_csv('datasets/'+df_name)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    automl = TabularAutoML(
        task=Task(
            name=task_type,
            metric=task_metric
        ),
        timeout=30
    )

    oof_preds = automl.fit_predict(train_df, roles={'target': target}).data
    test_preds = automl.predict(test_df).data
    if task_type == "reg":
        print("R2 score on oof data:", r2_score(train_df[target].values, oof_preds[:, 0]))
        print("R2 score on test data:", r2_score(test_df[target].values, test_preds[:, 0]))
    else:
        print("ROC-AUC score on oof data:", roc_auc_score(train_df[target].values, oof_preds[:, 0]))
        print("ROC-AUC score on test data:", roc_auc_score(test_df[target].values, test_preds[:, 0]))


if __name__ == '__main__':
    main()
