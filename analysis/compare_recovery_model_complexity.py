from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "experimental_data.csv"
OUT_DIR = ROOT / "analysis"

PHI_COL = "porosity_pct"
K_COL = "permeability_mD"
P_COL = "pressure_psi"
STAGE_COL = "stage"
TIME_COL = "rest_time_hr"

RECOVERY_ANCHORS = {
    "rest_after_5000psi": "5000psi_loading",
    "rest_after_7000psi": "7000psi_loading",
}


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - observed
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator > 0:
        r2 = float(
            1.0 - np.sum(errors**2) / denominator
        )
    else:
        r2 = float("nan")

    return {
        "N": len(observed),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def fit_predict(train, test, model_name):
    tau_train = np.log1p(
        train[TIME_COL].to_numpy(dtype=float)
    )
    tau_test = np.log1p(
        test[TIME_COL].to_numpy(dtype=float)
    )

    z_train = train["pressure_norm"].to_numpy(dtype=float)
    z_test = test["pressure_norm"].to_numpy(dtype=float)

    phi_anchor = test["phi_anchor_pct"].to_numpy(dtype=float)
    k_anchor = test["k_anchor_mD"].to_numpy(dtype=float)

    if model_name == "anchor_baseline":
        delta_phi = np.zeros(len(test))
        log_k_ratio = np.zeros(len(test))

    else:
        if model_name == "time_only":
            X_train = tau_train[:, None]
            X_test = tau_test[:, None]

        elif model_name == "time_pressure":
            X_train = np.column_stack(
                [tau_train, z_train * tau_train]
            )
            X_test = np.column_stack(
                [tau_test, z_test * tau_test]
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_phi = (
            train[PHI_COL].to_numpy(dtype=float)
            - train["phi_anchor_pct"].to_numpy(dtype=float)
        )

        y_logk = np.log(
            train[K_COL].to_numpy(dtype=float)
            / train["k_anchor_mD"].to_numpy(dtype=float)
        )

        beta_phi, *_ = np.linalg.lstsq(
            X_train, y_phi, rcond=None
        )
        beta_logk, *_ = np.linalg.lstsq(
            X_train, y_logk, rcond=None
        )

        delta_phi = X_test @ beta_phi
        log_k_ratio = X_test @ beta_logk

    return (
        phi_anchor + delta_phi,
        k_anchor * np.exp(log_k_ratio),
    )


def main():
    df = pd.read_csv(DATA_PATH)

    required = {
        P_COL, PHI_COL, K_COL, STAGE_COL, TIME_COL
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in [P_COL, PHI_COL, K_COL, TIME_COL]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if (df[K_COL] <= 0).any():
        raise ValueError("Permeability must be positive.")

    p_min = df[P_COL].min()
    p_max = df[P_COL].max()

    if p_max <= p_min:
        raise ValueError("Pressure range is invalid.")

    df["pressure_norm"] = (
        (df[P_COL] - p_min) / (p_max - p_min)
    )

    all_predictions = []
    summary_rows = []

    model_names = [
        "anchor_baseline",
        "time_only",
        "time_pressure",
    ]

    for recovery_stage, anchor_stage in RECOVERY_ANCHORS.items():
        recovery = df.loc[
            df[STAGE_COL] == recovery_stage
        ].copy()

        anchor = df.loc[
            df[STAGE_COL] == anchor_stage,
            [P_COL, PHI_COL, K_COL],
        ].copy()

        anchor = anchor.rename(
            columns={
                PHI_COL: "phi_anchor_pct",
                K_COL: "k_anchor_mD",
            }
        )

        if anchor[P_COL].duplicated().any():
            raise ValueError(
                f"Duplicate anchor pressures in {anchor_stage}"
            )

        recovery = recovery.merge(
            anchor,
            on=P_COL,
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        if (recovery["_merge"] != "both").any():
            raise ValueError(
                f"Some pressures in {recovery_stage} "
                "have no matching loading anchor."
            )

        recovery = recovery.drop(columns="_merge")

        if recovery.empty:
            raise ValueError(
                f"No recovery data found for {recovery_stage}"
            )

        times = sorted(recovery[TIME_COL].unique())

        print(
            f"\nStage: {recovery_stage} | "
            f"Observations: {len(recovery)} | "
            f"Time levels: {times}"
        )

        for held_time in times:
            train = recovery.loc[
                recovery[TIME_COL] != held_time
            ].copy()

            test = recovery.loc[
                recovery[TIME_COL] == held_time
            ].copy()

            if train.empty or test.empty:
                raise ValueError(
                    f"Invalid fold at time {held_time}"
                )

            for model_name in model_names:
                phi_pred, k_pred = fit_predict(
                    train, test, model_name
                )

                fold = test.copy()
                fold["held_out_time_hr"] = held_time
                fold["validation_type"] = (
                    "leave_one_time_level_out"
                )
                fold["model"] = model_name
                fold["phi_predicted_pct"] = phi_pred
                fold["k_predicted_mD"] = k_pred

                all_predictions.append(fold)

    predictions = pd.concat(
        all_predictions, ignore_index=True
    )

    for stage in RECOVERY_ANCHORS:
        stage_data = predictions.loc[
            predictions["stage"] == stage
        ]

        for variable, observed_col, predicted_col in [
            (
                "porosity",
                PHI_COL,
                "phi_predicted_pct",
            ),
            (
                "permeability",
                K_COL,
                "k_predicted_mD",
            ),
        ]:
            for model_name in model_names:
                part = stage_data.loc[
                    stage_data["model"] == model_name
                ]

                result = regression_metrics(
                    part[observed_col],
                    part[predicted_col],
                )

                summary_rows.append({
                    "stage": stage,
                    "variable": variable,
                    "model": model_name,
                    **result,
                })

    summary = pd.DataFrame(summary_rows)

    predictions_path = (
        OUT_DIR / "recovery_model_complexity_predictions.csv"
    )
    summary_path = (
        OUT_DIR / "recovery_model_complexity_comparison.csv"
    )

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPOOLED HOLDOUT COMPARISON")
    print(
        summary.sort_values(
            ["stage", "variable", "RMSE"]
        ).to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\nSAVED FILES")
    print(predictions_path)
    print(summary_path)

    print("\nINTERPRETATION")
    print(
        "Lower RMSE and MAE indicate smaller prediction errors. "
        "Higher R2 is generally preferable."
    )
    print(
        "The folds hold out complete observed recovery-time levels, "
        "but reuse the same samples and pressure levels."
    )
    print(
        "This comparison is empirical model selection, not "
        "independent validation on new rock samples."
    )
    print(
        "Only three observed recovery-time levels are available "
        "per stage, so conclusions about model complexity are preliminary."
    )


if __name__ == "__main__":
    main()
