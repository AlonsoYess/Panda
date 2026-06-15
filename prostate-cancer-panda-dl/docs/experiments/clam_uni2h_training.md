# Entrenamiento CLAM + UNI2-h

## Objetivo

Este experimento implementa el segundo modelo oficial del proyecto:
`CLAM + UNI2-h` para deteccion binaria de cancer de prostata a nivel WSI.

Entrada:
- embeddings UNI2-h previamente generados.

Salida:
- modelo CLAM binario para `cancer_label`.

Este flujo no vuelve a procesar imagenes crudas, no lee ZIPs y no extrae nuevos
embeddings. Solo consume los `.pt` ya validados.

## Rutas

Embeddings de entrada:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/embeddings
```

Outputs CLAM:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/clam_uni2h_binary
```

CLAM solo lee embeddings desde el output de ABMIL + UNI2-h. No modifica
checkpoints, metricas ni graficos de ABMIL.

## ABMIL vs CLAM

ABMIL usa atencion para ponderar tiles y agregarlos en una representacion de
bolsa/WSI.

CLAM tambien usa atencion, pero agrega supervision instance-level: en bolsas
positivas usa tiles de alta atencion como instancias positivas y tiles de baja
atencion como negativas; en bolsas negativas usa tiles de alta atencion como
negativas. Esto puede mejorar la separacion de regiones relevantes.

## Dry-run

```bash
python scripts/13_train_clam_uni2h_binary.py \
  --config configs/clam_uni2h_train_binary.yaml \
  --dry-run \
  --device cuda
```

## Smoke test

```bash
python scripts/13_train_clam_uni2h_binary.py \
  --config configs/clam_uni2h_train_binary.yaml \
  --epochs 2 \
  --max-train 64 \
  --max-valid 32 \
  --device cuda \
  --no-resume
```

## Entrenamiento oficial

```bash
python scripts/13_train_clam_uni2h_binary.py \
  --config configs/clam_uni2h_train_binary.yaml \
  --device cuda \
  --no-resume
```

## Resume si Colab se corta

```bash
python scripts/13_train_clam_uni2h_binary.py \
  --config configs/clam_uni2h_train_binary.yaml \
  --device cuda \
  --resume
```

El entrenamiento guarda `last_checkpoint.pt` al cierre de cada epoca y reanuda
desde la epoca siguiente.

## Evaluacion oficial

```bash
python scripts/14_evaluate_clam_uni2h_binary.py \
  --config configs/clam_uni2h_train_binary.yaml \
  --device cuda
```

La evaluacion calcula metricas con threshold `0.5`, selecciona threshold Youden
en validacion y aplica ese threshold al test.

## Outputs

```text
checkpoints/best_model.pt
checkpoints/last_checkpoint.pt
checkpoints/checkpoint_epoch_XX.pt
metrics/train_history.csv
metrics/run_metadata.json
metrics/valid_metrics.json
metrics/test_metrics.json
metrics/best_threshold.json
metrics/valid_predictions.csv
metrics/test_predictions.csv
plots/confusion_matrix_valid.png
plots/confusion_matrix_test.png
plots/roc_valid.png
plots/roc_test.png
plots/training_loss.png
plots/validation_auc_f1.png
```

No subir embeddings, checkpoints, metricas ni graficos pesados a GitHub.
