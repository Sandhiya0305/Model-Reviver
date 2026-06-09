import pandas as pd
from sklearn.preprocessing import LabelEncoder
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

import config as cfg


def _prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "bool"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.select_dtypes(include=["number"]).fillna(0)
    return df


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref = _prepare_columns(reference)
    cur = _prepare_columns(current)

    overlap = [c for c in ref.columns if c in cur.columns]
    if not overlap:
        return {"drift_detected": False, "drift_score": 0.0, "error": "no overlapping columns"}

    ref = ref[overlap]
    cur = cur[overlap]

    column_mapping = ColumnMapping(
        numerical_features=[c for c in ref.columns if c != "label"],
        target="label" if "label" in ref.columns else None,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    results = report.as_dict()
    metrics = results.get("metrics", [])
    drift_share = 0.0
    for m in metrics:
        if m.get("metric", "") in ("DatasetDriftMetric", "DataDriftPreset"):
            drift_share = m.get("result", {}).get("share_of_drifted_columns", 0.0)
            break

    return {
        "drift_detected": drift_share >= cfg.DRIFT_THRESHOLD,
        "drift_score": drift_share,
        "reference_rows": len(ref),
        "current_rows": len(cur),
    }
