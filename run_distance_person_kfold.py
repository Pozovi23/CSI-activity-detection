#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from csi_wifi_common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SINC_FILTERS,
    DEFAULT_SINC_KERNEL_SIZE,
    DEFAULT_STEP,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WINDOW,
    DISTANCE_CLASS_NAMES,
    apply_normalizer,
    build_distance_windows,
    build_recording_index,
    evaluate_distance,
    filter_activity_summary_by_class,
    fit_normalizer,
    infer_null_subcarriers,
    make_loader,
    reshape_for_model,
    set_seed,
    split_recordings,
    summarize_recording_overlap,
    train_distance_model,
)


def main() -> None:
    set_seed(DEFAULT_RANDOM_STATE)
    recordings_df = build_recording_index()
    rows = []

    for fold_idx, test_person in enumerate(sorted(recordings_df["person"].unique()), start=1):
        print(f"\n=== Distance fold {fold_idx}: test_person={test_person} ===")

        test_recordings = recordings_df[recordings_df["person"] == test_person].reset_index(drop=True)
        train_val_recordings = recordings_df[recordings_df["person"] != test_person].reset_index(drop=True)
        train_recordings, val_recordings, _ = split_recordings(
            train_val_recordings,
            group_col="recording_id",
            test_size=0.0,
            val_size=0.25,
            random_state=DEFAULT_RANDOM_STATE + fold_idx,
        )

        overlap = summarize_recording_overlap(train_recordings, val_recordings, test_recordings)
        assert overlap["recording_train_test"] == 0
        assert overlap["person_train_test"] == 0

        null_subcarriers = infer_null_subcarriers(train_recordings)

        X_train, y_train, _, train_meta = build_distance_windows(
            train_recordings,
            null_subcarriers,
            window=DEFAULT_WINDOW,
            step=DEFAULT_STEP,
            preprocess_config=DEFAULT_PREPROCESS_CONFIG,
            group_col="recording_id",
            desc=f"Build train windows fold {fold_idx}",
        )
        X_val, y_val, _, val_meta = build_distance_windows(
            val_recordings,
            null_subcarriers,
            window=DEFAULT_WINDOW,
            step=DEFAULT_STEP,
            preprocess_config=DEFAULT_PREPROCESS_CONFIG,
            group_col="recording_id",
            desc=f"Build val windows fold {fold_idx}",
        )
        X_test, y_test, _, test_meta = build_distance_windows(
            test_recordings,
            null_subcarriers,
            window=DEFAULT_WINDOW,
            step=DEFAULT_STEP,
            preprocess_config=DEFAULT_PREPROCESS_CONFIG,
            group_col="person",
            desc=f"Build test windows fold {fold_idx}",
        )

        X_train = reshape_for_model(X_train)
        X_val = reshape_for_model(X_val)
        X_test = reshape_for_model(X_test)

        mean, std = fit_normalizer(X_train)
        X_train = apply_normalizer(X_train, mean, std)
        X_val = apply_normalizer(X_val, mean, std)
        X_test = apply_normalizer(X_test, mean, std)

        model, device = train_distance_model(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            lr=DEFAULT_LR,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            sinc_filters=DEFAULT_SINC_FILTERS,
            sinc_kernel_size=DEFAULT_SINC_KERNEL_SIZE,
        )

        test_loader = make_loader(X_test, y_test.astype(int), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
        test_metrics = evaluate_distance(model, test_loader, device)
        filter_report = filter_activity_summary_by_class(model, X_test, y_test, device, DISTANCE_CLASS_NAMES)

        rows.append(
            {
                "fold": fold_idx,
                "test_person": test_person,
                "test_acc": test_metrics["accuracy"],
                "test_bal_acc": test_metrics["balanced_accuracy"],
                "test_mae": test_metrics["mae"],
                "test_windows": int(len(y_test)),
            }
        )

        print(
            {
                "train_windows": len(train_meta),
                "val_windows": len(val_meta),
                "test_windows": len(test_meta),
                "person_overlap_train_test": overlap["person_train_test"],
                "test_distance_counts": test_meta["distance_m"].value_counts().sort_index().to_dict(),
            }
        )
        print(
            f"acc={test_metrics['accuracy']:.4f} "
            f"bal_acc={test_metrics['balanced_accuracy']:.4f} "
            f"mae={test_metrics['mae']:.4f}"
        )
        print(test_metrics["report"])
        print(filter_report.to_string(index=False))

    results = pd.DataFrame(rows)
    print("\n=== Distance Person K-Fold Summary ===")
    print(results.to_string(index=False))
    print(
        {
            "mean_test_bal_acc": round(float(results["test_bal_acc"].mean()), 4),
            "std_test_bal_acc": round(float(results["test_bal_acc"].std(ddof=0)), 4),
            "mean_test_mae": round(float(results["test_mae"].mean()), 4),
        }
    )


if __name__ == "__main__":
    main()
