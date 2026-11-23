
# ============================================================
# STUDENT EXPERIMENT
# Comparison of Grad-CAM: ResNet18 Layer3 vs Layer4
# Dataset: NEU-DET
# ============================================================

from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets

# Reuse the original repository's implementation
from create_gradcam_examples import (
    GradCAM,
    build_model,
    build_transform,
    collect_balanced_examples,
    create_input_tensor,
    load_display_image,
    resize_heatmap,
)

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/NEU-DET")
MODEL_PATH = Path("models/resnet18_best.pth")
RESULTS_DIR = Path("results")

EXAMPLES_PER_CLASS = 2
IMAGE_SIZE = 224

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = RESULTS_DIR / "gradcam_layer3_vs_layer4.png"
CSV_PATH = RESULTS_DIR / "gradcam_layer_comparison.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

# ============================================================
# STEP 2: LOAD VALIDATION DATASET
# ============================================================

transform = build_transform()

dataset = datasets.ImageFolder(
    root=DATA_DIR / "validation" / "images",
    transform=transform
)

class_names = dataset.classes

print("Classes:", class_names)
print("Validation images:", len(dataset))

examples = collect_balanced_examples(
    dataset=dataset,
    class_names=class_names,
    examples_per_class=EXAMPLES_PER_CLASS
)

print("Selected examples:", len(examples))

# ============================================================
# STEP 3: LOAD TRAINED RESNET18
# ============================================================

model = build_model(
    num_classes=len(class_names)
).to(device)

model.eval()

print("Trained ResNet18 loaded successfully.")

# ============================================================
# STEP 4: SELECT TARGET LAYERS
# ============================================================

target_layers = {
    "Layer3": model.layer3[-1],
    "Layer4": model.layer4[-1],
}

# ============================================================
# STEP 5: CREATE VISUALIZATION
# ============================================================

num_rows = len(examples)
num_columns = 5

fig, axes = plt.subplots(
    num_rows,
    num_columns,
    figsize=(17, num_rows * 2.8),
    squeeze=False
)

csv_results = []

for row, (image_path, true_idx) in enumerate(examples):

    input_tensor = create_input_tensor(
        image_path=image_path,
        transform=transform,
        device=device
    )

    original = load_display_image(image_path)

    # Predict once with the same checkpoint
    with torch.no_grad():
        prediction_output = model(input_tensor)
        probs = torch.softmax(prediction_output, dim=1)
        confidence, pred_tensor = probs.max(dim=1)

    pred_idx = int(pred_tensor.item())
    conf = float(confidence.item())

    true_label = class_names[true_idx]
    pred_label = class_names[pred_idx]

    correct = pred_idx == true_idx

    # First column: original image
    axes[row, 0].imshow(original, cmap="gray")
    axes[row, 0].set_title(
        f"Original\nTrue: {true_label}",
        fontsize=9
    )

    record = {
        "image": image_path.name,
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": conf,
        "correct": correct,
    }

    # Grad-CAM for each target layer
    for layer_index, (layer_name, layer) in enumerate(
        target_layers.items()
    ):

        gradcam = GradCAM(
            model=model,
            target_layer=layer
        )

        try:
            # Explain the same predicted class
            cam, _ = gradcam.generate(
                input_tensor=input_tensor,
                target_class=pred_idx
            )
        finally:
            gradcam.close()

        heatmap = resize_heatmap(
            cam,
            image_size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        heatmap_column = 1 + (layer_index * 2)
        overlay_column = heatmap_column + 1

        # Heatmap
        axes[row, heatmap_column].imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1
        )

        axes[row, heatmap_column].set_title(
            f"{layer_name} Heatmap",
            fontsize=9
        )

        # Overlay
        axes[row, overlay_column].imshow(
            original,
            cmap="gray"
        )

        axes[row, overlay_column].imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1
        )

        axes[row, overlay_column].set_title(
            f"{layer_name} Overlay\n"
            f"Pred: {pred_label} | Conf: {conf:.2f}",
            fontsize=8
        )

        # Descriptive measure only:
        # fraction of pixels with normalized CAM >= 0.75
        record[
            f"{layer_name.lower()}_high_activation_fraction"
        ] = float(np.mean(heatmap >= 0.75))

    csv_results.append(record)

    for ax in axes[row]:
        ax.axis("off")

    status = "Correct" if correct else "Wrong"

    print(
        f"{image_path.name} | "
        f"True: {true_label} | "
        f"Pred: {pred_label} | "
        f"{status}"
    )

# ============================================================
# STEP 6: SAVE VISUALIZATION
# ============================================================

fig.suptitle(
    "ResNet18 Grad-CAM Comparison: Layer3 vs Layer4",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.985])

plt.savefig(
    FIGURE_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# STEP 7: SAVE CSV
# ============================================================

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(csv_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(csv_results)

print("\nExperiment completed.")
print("Figure:", FIGURE_PATH)
print("CSV:", CSV_PATH)
