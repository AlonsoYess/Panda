# Entrenamiento oficial ABMIL + UNI2-h

## Objetivo

Este experimento entrena ABMIL para clasificacion binaria de cancer de prostata
a nivel WSI:

- `cancer_label = 0`: no cancer
- `cancer_label = 1`: cancer

ABMIL recibe exclusivamente embeddings congelados de `MahmoodLab/UNI2-h` con
1536 dimensiones. No vuelve a abrir WSI, procesar imagenes ni extraer features.

## Datos validados

Ruta esperada:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/embeddings
```

Distribucion validada:

| Split | WSI | No cancer | Cancer |
|---|---:|---:|---:|
| train | 7430 | 2024 | 5406 |
| valid | 1592 | 434 | 1158 |
| test | 1077 | 309 | 768 |
| total | 10099 | 2767 | 7332 |

Cada archivo debe declarar `encoder_family=UNI2-h`, dimensión `1536` y features
`[num_tiles, 1536]`. Los embeddings UNI clasico de 1024 dimensiones se rechazan.

## Dry-run

El dry-run valida train/valid, construye ABMIL y ejecuta un forward de una WSI.
No entrena y no escribe checkpoints:

```bash
python scripts/11_train_abmil_uni2h_binary.py \
  --config configs/abmil_uni2h_train_binary.yaml \
  --dry-run \
  --max-train 5 \
  --max-valid 5
```

## Entrenamiento corto

```bash
python scripts/11_train_abmil_uni2h_binary.py \
  --config configs/abmil_uni2h_train_binary.yaml \
  --epochs 2 \
  --max-train 100 \
  --max-valid 50 \
  --output-root /content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary_smoke \
  --no-resume
```

Este comando usa un subconjunto y debe considerarse una prueba tecnica, no un
resultado científico final. La carpeta `abmil_uni2h_binary_smoke` evita mezclar
sus checkpoints con el entrenamiento oficial.

## Entrenamiento completo

```bash
python scripts/11_train_abmil_uni2h_binary.py \
  --config configs/abmil_uni2h_train_binary.yaml
```

El entrenamiento utiliza únicamente `train` y `valid`. El split `test` no se
carga ni evalúa durante esta etapa.

## Reanudar despues de una desconexion

Al terminar cada época se guardan en Google Drive:

- `last_checkpoint.pt`
- `train_history.csv`
- `checkpoint_epoch_XX.pt`, si `save_every_epoch=true`
- `best_model.pt`, cuando mejora la métrica monitorizada

Para reanudar:

```bash
python scripts/11_train_abmil_uni2h_binary.py \
  --config configs/abmil_uni2h_train_binary.yaml \
  --resume
```

Si `last_checkpoint.pt` terminó en la época 18, la nueva sesión continúa desde
la época 19. También se restauran optimizador, GradScaler, historial, mejor
métrica y contador de early stopping.

`last_checkpoint.pt` representa el último estado recuperable. `best_model.pt`
representa la época con mejor AUC de validación y es el archivo usado para la
evaluación final.

## Evaluación final

Después de completar el entrenamiento:

```bash
python scripts/12_evaluate_abmil_uni2h_binary.py \
  --config configs/abmil_uni2h_train_binary.yaml
```

El umbral de Youden se calcula exclusivamente con validación. Después se aplica
sin cambios al test. El test nunca interviene en selección de modelo, early
stopping ni selección de umbral.

## Outputs

Checkpoints:

```text
checkpoints/last_checkpoint.pt
checkpoints/best_model.pt
checkpoints/checkpoint_epoch_XX.pt
```

Métricas:

```text
metrics/run_metadata.json
metrics/train_history.csv
metrics/valid_predictions.csv
metrics/test_predictions.csv
metrics/valid_metrics.json
metrics/test_metrics.json
metrics/best_threshold.json
```

Gráficos:

```text
plots/training_loss.png
plots/validation_auc_f1.png
plots/confusion_matrix_valid.png
plots/confusion_matrix_test.png
plots/roc_valid.png
plots/roc_test.png
```

## Interpretación

- `AUC ROC`: capacidad de ordenar WSI con y sin cáncer, independiente del umbral.
- `Recall/Sensitivity`: proporción de casos con cáncer detectados.
- `Specificity`: proporción de casos sin cáncer correctamente descartados.
- `F1`: equilibrio entre precision y sensitivity.
- `Gini`: `2 * AUC - 1`.
- `Confusion matrix`: conteos TN, FP, FN y TP.

Todos los embeddings, checkpoints, métricas y gráficos pesados permanecen en
Google Drive. No deben subirse al repositorio Git.
