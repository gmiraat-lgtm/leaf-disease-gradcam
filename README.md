# M2 — Leaf Disease Detection with Grad-CAM

**Smart Crop Advisory System (SIH25010)** · B.Tech Project, SEM VII · Symbiosis Institute of Technology, Pune — Dept. of AI & ML

**Module owner:** Miraat Gupta (23070126073)

Identifies crop disease from a single leaf photograph across all 38 PlantVillage classes, and shows *why*: every prediction returns a Grad-CAM heatmap of the leaf region that drove the decision, an indicative severity level, and a matching knowledge-base entry (symptoms, the key visual cue to verify where the heatmap activates, and the recommended action), so a non-expert can cross-check the model instead of trusting a bare label.

## Notebooks

| Notebook | What it is |
|---|---|
| `M2_Leaf_Disease_GradCAM.ipynb` | Baseline — MobileNetV2 transfer learning on a 10-class subset, Grad-CAM, Gradio app. Establishes the measurable reference point (98.17% validation accuracy). |
| `M2_Main_38class_TwoModels.ipynb` | Main model — all 38 classes, two backbones trained under an identical protocol (MobileNetV2 and EfficientNet-B0), Drive-checkpointed resumable training, model comparison, Grad-CAM for both, 38-entry disease knowledge base, and a standalone Gradio deployment cell. |

## Results (38 classes, 20% validation split)

| Model | Validation accuracy |
|---|---|
| MobileNetV2 | 98.33% |
| EfficientNet-B0 | **99.33%** |

Two-phase transfer learning (frozen-backbone head training, then fine-tuning of the top layers at a low learning rate with BatchNormalization kept frozen), class weights for the 152-vs-5,500 image imbalance, EarlyStopping with best-weight restoration, and `ModelCheckpoint` writing to Google Drive after every improving epoch so interrupted Colab sessions resume instead of restarting.

## How to run (Google Colab)

1. Open `M2_Main_38class_TwoModels.ipynb` in Colab (GPU runtime for training; CPU is enough for the app).
2. Run All — it mounts Google Drive, downloads the dataset (needs your own `kaggle.json`), and trains both models with checkpointing. Training cells are gated: if a finished model already exists in Drive they skip themselves.
3. For every later session, run only the final standalone cell — it loads the saved models from Drive and launches the Gradio app with a public share link.

## Dataset

[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 54,305 images, 38 classes. The Kaggle page mirrors the dataset of the PlantVillage project (Penn State University & EPFL); labels derive from supervised inoculation field trials at US Land Grant University research stations (Hughes & Salathé, arXiv:1511.08060; Mohanty, Hughes & Salathé, *Frontiers in Plant Science* 7:1419, 2016).

**Stated limitation:** PlantVillage is laboratory imagery with uniform backgrounds — reported accuracy overstates in-field performance, and the severity level is a heatmap-area heuristic, not a calibrated agronomic measurement. This is a research prototype, not a substitute for professional agronomic advice.
