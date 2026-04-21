#!/usr/bin/env python3
from __future__ import annotations

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
    train_recordings, val_recordings, test_recordings = split_recordings(
        recordings_df,
        group_col="recording_id",
        random_state=DEFAULT_RANDOM_STATE,
    )

    overlap = summarize_recording_overlap(train_recordings, val_recordings, test_recordings)
    assert overlap["recording_train_val"] == 0
    assert overlap["recording_train_test"] == 0
    assert overlap["recording_val_test"] == 0

    null_subcarriers = infer_null_subcarriers(train_recordings)

    X_train, y_train, _, train_meta = build_distance_windows(
        train_recordings,
        null_subcarriers,
        window=DEFAULT_WINDOW,
        step=DEFAULT_STEP,
        preprocess_config=DEFAULT_PREPROCESS_CONFIG,
        group_col="recording_id",
        desc="Build train windows",
    )
    X_val, y_val, _, val_meta = build_distance_windows(
        val_recordings,
        null_subcarriers,
        window=DEFAULT_WINDOW,
        step=DEFAULT_STEP,
        preprocess_config=DEFAULT_PREPROCESS_CONFIG,
        group_col="recording_id",
        desc="Build val windows",
    )
    X_test, y_test, _, test_meta = build_distance_windows(
        test_recordings,
        null_subcarriers,
        window=DEFAULT_WINDOW,
        step=DEFAULT_STEP,
        preprocess_config=DEFAULT_PREPROCESS_CONFIG,
        group_col="recording_id",
        desc="Build test windows",
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

    val_loader = make_loader(X_val, y_val.astype(int), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test.astype(int), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    val_metrics = evaluate_distance(model, val_loader, device)
    test_metrics = evaluate_distance(model, test_loader, device)
    filter_report = filter_activity_summary_by_class(model, X_test, y_test, device, DISTANCE_CLASS_NAMES)

    print("\nLeak check:")
    print(overlap)

    print("\nSplit sizes:")
    print(
        {
            "train_recordings": len(train_recordings),
            "val_recordings": len(val_recordings),
            "test_recordings": len(test_recordings),
            "train_windows": len(train_meta),
            "val_windows": len(val_meta),
            "test_windows": len(test_meta),
            "train_distance_counts": train_meta["distance_m"].value_counts().sort_index().to_dict(),
            "val_distance_counts": val_meta["distance_m"].value_counts().sort_index().to_dict(),
            "test_distance_counts": test_meta["distance_m"].value_counts().sort_index().to_dict(),
        }
    )

    print(f"\nval_acc={val_metrics['accuracy']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} val_mae={val_metrics['mae']:.4f}")
    print(val_metrics["report"])
    print(f"test_acc={test_metrics['accuracy']:.4f} test_bal_acc={test_metrics['balanced_accuracy']:.4f} test_mae={test_metrics['mae']:.4f}")
    print(test_metrics["report"])

    print("\nLearned sinc filters:")
    print(filter_report.to_string(index=False))


if __name__ == "__main__":
    main()
