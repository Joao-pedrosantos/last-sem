"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
        remainder="drop",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train, model_type="knn"):
    """
    Train a classification model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target
        model_type (str): Model type ("logistic" or "knn")

    Returns:
        sklearn classifier: trained model
    """
    if model_type == "logistic":
        max_iter = 1000
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_iter", max_iter)
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)
    else:
        mlflow.log_param("model_type", "knn")
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
        }
        search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        mlflow.log_param("best_n_neighbors", search.best_params_["n_neighbors"])
        mlflow.log_param("best_weights", search.best_params_["weights"])
        mlflow.log_metric("cv_best_f1", search.best_score_)

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train.head(3),
    )
    return model


def main():
    mlflow.set_experiment("churn-exp")
    with mlflow.start_run():
        run_name = "knn-gridsearch-with-scaler"
        mlflow.set_tag("mlflow.runName", run_name)

        df = pd.read_csv("data/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        mlflow.log_param("feature_names", ", ".join(X_train.columns))

        model = train(X_train, y_train, model_type="knn")
        y_pred = model.predict(X_test)

        print(f"Accuracy score: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision score: {precision_score(y_test, y_pred):.2f}")
        print(f"Recall score: {recall_score(y_test, y_pred):.2f}")
        print(f"F1 score: {f1_score(y_test, y_pred):.2f}")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.sklearn.log_model(col_transf, "col_transf")
        mlflow.log_artifact("data/Churn_Modelling.csv", "data")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        fig, ax = plt.subplots()
        conf_mat_disp.plot(ax=ax)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        mlflow.log_image(image, "confusion_matrix.png")
        plt.close(fig)


if __name__ == "__main__":
    main()