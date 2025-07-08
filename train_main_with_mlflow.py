import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import mlflow
import mlflow.sklearn
import numpy as np
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MLflow 설정 ---
mlflow_tracking_uri = "http://localhost:5050"
mlflow.set_tracking_uri(mlflow_tracking_uri)
os.environ["MLFLOW_REGISTRY_URI"] = mlflow_tracking_uri

# 전처리 파라미터 (학습과 서빙에서 일관성 유지를 위해 저장)
PREPROCESSING_PARAMS = {
    "age_median": None,
    "fare_median": None,
    "embarked_mode": None,
    "train_columns": [],  # 학습 데이터의 최종 컬럼 순서 및 이름 저장
}


def preprocess_data(df, is_training=True):
    """
    데이터 전처리 함수: 결측치 처리, 인코딩, 특성 공학
    is_training=True일 경우 전처리 파라미터(중앙값, 최빈값 등)를 계산하고 저장합니다.
    is_training=False일 경우 저장된 파라미터를 사용하여 전처리를 수행합니다.
    """
    global PREPROCESSING_PARAMS

    # Age 결측치: 중앙값 대체
    if is_training:
        PREPROCESSING_PARAMS["age_median"] = df["Age"].median()
    df["Age"] = df["Age"].fillna(PREPROCESSING_PARAMS["age_median"])

    # Fare 결측치: 중앙값 대체
    if is_training:
        PREPROCESSING_PARAMS["fare_median"] = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(PREPROCESSING_PARAMS["fare_median"])

    # Embarked 결측치: 최빈값 대체
    if is_training:
        PREPROCESSING_PARAMS["embarked_mode"] = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(PREPROCESSING_PARAMS["embarked_mode"])

    # Sex 인코딩
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Embarked 원-핫 인코딩
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # FamilySize 특성 공학
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Age/Fare 구간화 (labels와 bins는 학습 시 결정된 값을 사용해야 함)
    age_bins = [0.17, 16.16, 32.16, 48.16, 64.16, 80.16]
    age_labels = [0, 1, 2, 3, 4]
    df["AgeBand"] = pd.cut(
        df["Age"], bins=age_bins, labels=age_labels, right=True, include_lowest=True
    )
    df = df.drop("Age", axis=1)

    fare_qs = [0, 0.25, 0.5, 0.75, 1]
    df["FareBand"] = pd.qcut(
        df["Fare"], q=fare_qs, labels=[0, 1, 2, 3], duplicates="drop"
    )
    df = df.drop("Fare", axis=1)

    # 불필요한 원본 컬럼 제거 (SibSp, Parch는 FamilySize로 대체)
    cols_to_drop_after_process = ["Name", "Ticket", "Cabin", "SibSp", "Parch"]
    for col in cols_to_drop_after_process:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # 모델 학습 시 사용한 최종 컬럼 순서 및 이름 맞추기 (매우 중요!)
    if is_training:
        PREPROCESSING_PARAMS["train_columns"] = df.columns.tolist()
    else:
        expected_cols = PREPROCESSING_PARAMS.get("train_columns", [])
        if not expected_cols:
            logger.warning(
                "train_columns이 PREPROCESSING_PARAMS에 없습니다. 수동으로 기본 컬럼을 사용합니다."
            )
            expected_cols = [
                "Pclass",
                "Sex",
                "Embarked_Q",
                "Embarked_S",
                "FamilySize",
                "AgeBand",
                "FareBand",
            ]

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]

    return df


def train_model_main():
    """간소화된 모델 학습 및 MLflow 로깅"""
    with mlflow.start_run(run_name="Titanic_Main_Logistic_Regression_Training"):
        logger.info("MLflow 실험 시작 (간소화 버전)")

        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        test_passenger_id = test_df["PassengerId"]

        X_train_raw = train_df.drop(["Survived", "PassengerId"], axis=1)
        y_train_raw = train_df["Survived"]
        processed_train_df = preprocess_data(X_train_raw.copy(), is_training=True)

        X_train, X_val, y_train, y_val = train_test_split(
            processed_train_df,
            y_train_raw,
            test_size=0.2,
            random_state=42,
            stratify=y_train_raw,
        )
        logger.info(f"훈련 데이터셋 크기: {X_train.shape}")
        logger.info(f"검증 데이터셋 크기: {X_val.shape}")

        model = LogisticRegression(solver="liblinear", random_state=42)
        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred_val)
        precision = precision_score(y_val, y_pred_val)
        recall = recall_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val)
        roc_auc = roc_auc_score(y_val, y_pred_proba_val)

        logger.info(f"Accuracy (Validation): {accuracy:.4f}")
        logger.info(f"AUC (Validation): {roc_auc:.4f}")
        logger.info(
            f"Classification Report (Validation):\n{classification_report(y_val, y_pred_val)}"
        )
        logger.info(
            f"Confusion Matrix (Validation):\n{confusion_matrix(y_val, y_pred_val)}"
        )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        kfold_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "roc_auc": [],
        }

        logger.info("K-Fold Cross-Validation 시작...")
        for fold, (train_index, val_index) in enumerate(
            kf.split(processed_train_df, y_train_raw)
        ):
            X_train_fold, X_val_fold = (
                processed_train_df.iloc[train_index],
                processed_train_df.iloc[val_index],
            )
            y_train_fold, y_val_fold = (
                y_train_raw.iloc[train_index],
                y_train_raw.iloc[val_index],
            )

            fold_model = LogisticRegression(solver="liblinear", random_state=42)
            fold_model.fit(X_train_fold, y_train_fold)

            y_pred_fold = fold_model.predict(X_val_fold)
            y_pred_proba_fold = fold_model.predict_proba(X_val_fold)[:, 1]

            kfold_metrics["accuracy"].append(accuracy_score(y_val_fold, y_pred_fold))
            kfold_metrics["precision"].append(precision_score(y_val_fold, y_pred_fold))
            kfold_metrics["recall"].append(recall_score(y_val_fold, y_pred_fold))
            kfold_metrics["f1_score"].append(f1_score(y_val_fold, y_pred_fold))
            kfold_metrics["roc_auc"].append(
                roc_auc_score(y_val_fold, y_pred_proba_fold)
            )

        for metric_name, values in kfold_metrics.items():
            mlflow.log_metric(f"kfold_mean_{metric_name}", np.mean(values))
            mlflow.log_metric(f"kfold_std_{metric_name}", np.std(values))
        logger.info("K-Fold Cross-Validation 완료.")

        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_split_ratio", 0.2)
        mlflow.log_param("stratify_by_target", True)

        # 전처리 파라미터 저장
        preprocessor_state_path = "main_preprocessor_params.json"  # 파일명 변경
        with open(preprocessor_state_path, "w") as f:
            json.dump(PREPROCESSING_PARAMS, f)
        mlflow.log_artifact(preprocessor_state_path)
        os.remove(preprocessor_state_path)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # 모델 로깅 및 레지스트리 등록 (간소화 파이프라인)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="Titanic_Main_Logistic_Model",  # 모델 레지스트리 이름 변경
            signature=mlflow.models.signature.infer_signature(X_train, y_pred_val),
            input_example=X_train.head(1),
        )
        logger.info("모델 학습 및 MLflow 로깅 완료 (간소화 버전).")
        logger.info(
            f"모델 레지스트리에 'Titanic_Main_Logistic_Model' 이름으로 모델 등록."
        )

        X_test_raw = test_df.drop("PassengerId", axis=1)
        processed_test_df = preprocess_data(X_test_raw.copy(), is_training=False)

        final_predictions = model.predict(processed_test_df)

        submission_df = pd.DataFrame(
            {"PassengerId": test_passenger_id, "Survived": final_predictions}
        )
        submission_df.to_csv("submission_main.csv", index=False)  # 제출 파일명 변경
        logger.info("제출 파일 'submission_main.csv' 생성 완료.")


if __name__ == "__main__":
    train_model_main()
