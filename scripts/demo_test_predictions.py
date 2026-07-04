from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DEMO_DIR,
    EMOTION_NAMES,
    FIGURES_DIR,
    RANDOM_SEED,
)
from src.inference import (
    format_prediction_label,
    format_top_predictions,
    load_emotion_model,
    load_recommended_threshold,
    predict_emotion_from_face,
)
from src.processed_loader import load_processed_data


def select_demo_indices(
    number_of_samples: int,
    total_samples: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Select random test-set indices for demo visualization.
    """

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        total_samples,
        size=number_of_samples,
        replace=False,
    )

    return indices


def build_demo_figure(
    number_of_samples: int = 12,
) -> None:
    """
    Build and save a demo figure with model predictions on test images.
    """

    print("Loading selected model...")
    model = load_emotion_model()

    threshold = load_recommended_threshold()

    print(f"Using confidence threshold: {threshold:.2f}")

    print("Loading processed data...")
    splits = load_processed_data()

    indices = select_demo_indices(
        number_of_samples=number_of_samples,
        total_samples=len(splits.X_test),
    )

    rows = 3
    columns = 4

    if number_of_samples != rows * columns:
        raise ValueError(
            "number_of_samples must be 12 for the current 3x4 demo layout."
        )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 11),
    )

    axes = axes.flatten()

    for ax, index in zip(axes, indices):
        image = splits.X_test[index].squeeze()
        true_label_id = int(splits.y_test[index])
        true_emotion = EMOTION_NAMES[true_label_id]

        prediction = predict_emotion_from_face(
            model=model,
            face_image=image,
            threshold=threshold,
            use_histogram_equalization=False,
        )

        label = format_prediction_label(prediction)
        top_predictions = format_top_predictions(prediction)

        is_correct = (
            prediction.predicted_class_id == true_label_id
            and not prediction.is_rejected
        )

        if prediction.is_rejected:
            status = "Rejected"
        elif is_correct:
            status = "Correct"
        else:
            status = "Wrong"

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        title = (
            f"True: {true_emotion}\n"
            f"Pred: {label}\n"
            f"Status: {status}"
        )

        ax.set_title(
            title,
            fontsize=9,
        )

        ax.axis("off")

        top_text = "\n".join(top_predictions)

        ax.text(
            0.5,
            -0.15,
            top_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.suptitle(
        "FER2013 Test Set Predictions with Confidence Threshold",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()

    figure_path = FIGURES_DIR / "threshold_demo_test_predictions.png"
    demo_path = DEMO_DIR / "threshold_demo_test_predictions.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        demo_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Demo figure saved to: {figure_path}")
    print(f"Demo figure also saved to: {demo_path}")


def main() -> None:
    build_demo_figure(number_of_samples=12)


if __name__ == "__main__":
    main()