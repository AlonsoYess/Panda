# TransMIL + Virchow2 para deteccion binaria

## Objetivo

Este experimento entrena **TransMIL + Virchow2** para clasificacion binaria
de cancer de prostata usando embeddings Virchow2 previamente generados.

La etiqueta usada es:

- `cancer_label = 0`: no cancer
- `cancer_label = 1`: cancer

No se vuelven a procesar imagenes, tiles ni ZIPs. El entrenamiento lee
directamente archivos `.pt` de Virchow2.

## Entrada

Ruta esperada de embeddings:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings
```

Estructura esperada:

```text
embeddings/
|-- train/*.pt
|-- valid/*.pt
`-- test/*.pt
```

Contrato esperado por `.pt`:

- `features`: `torch.Tensor float32 [n_tiles, 1280]`
- `encoder_family`: `Virchow2`
- `embedding_dim`: `1280`
- `cancer_label`: `0` o `1`
- `split`: `train`, `valid` o `test`

## Salida

El experimento escribe solamente en:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/transmil_virchow2_binary
```

Outputs principales:

- `checkpoints/best_model.pt`
- `checkpoints/last_checkpoint.pt`
- `checkpoints/checkpoint_epoch_XX.pt`
- `logs/`
- `metrics/run_metadata.json`
- `metrics/train_history.csv`
- `metrics/valid_metrics.json`
- `metrics/test_metrics.json`
- `metrics/best_threshold.json`
- `metrics/valid_predictions.csv`
- `metrics/test_predictions.csv`
- `metrics/test_attention_scores.csv`
- `plots/training_loss.png`
- `plots/validation_auc_f1.png`
- `plots/roc_valid.png`
- `plots/roc_test.png`
- `plots/confusion_matrix_valid.png`
- `plots/confusion_matrix_test.png`

La evaluacion crea tambien:

```text
/content/drive/MyDrive/PANDA_PROSTATE/entregables/transmil_virchow2_binary_resultados
```

con:

- `metricas/`
- `graficas/`
- `modelo/`
- `regiones_relevantes/`
- `resumen_resultados_transmil_virchow2.json`

## Relacion con experimentos previos

Este flujo reutiliza la arquitectura TransMIL ya validada en TransMIL + UNI2-h,
pero cambia la entrada de `1536` a embeddings Virchow2 de `1280` dimensiones.

No modifica:

- TransMIL + UNI2-h
- ABMIL + Virchow2
- CLAM + Virchow2
- extraccion Virchow2

## Comandos Colab

### Dry-run

```bash
python scripts/25_train_transmil_virchow2_binary.py \
  --config configs/transmil_virchow2_train_binary.yaml \
  --dry-run \
  --device cuda
```

### Smoke test

```bash
python scripts/25_train_transmil_virchow2_binary.py \
  --config configs/transmil_virchow2_train_binary.yaml \
  --epochs 2 \
  --max-train-slides 64 \
  --max-valid-slides 32 \
  --device cuda \
  --output-root /content/drive/MyDrive/PANDA_PROSTATE/outputs/transmil_virchow2_binary_smoke \
  --no-resume
```

### Evaluacion smoke

```bash
python scripts/26_evaluate_transmil_virchow2_binary.py \
  --config configs/transmil_virchow2_train_binary.yaml \
  --device cuda \
  --output-root /content/drive/MyDrive/PANDA_PROSTATE/outputs/transmil_virchow2_binary_smoke \
  --max-valid 32 \
  --max-test 32
```

### Entrenamiento oficial

```bash
python scripts/25_train_transmil_virchow2_binary.py \
  --config configs/transmil_virchow2_train_binary.yaml \
  --device cuda \
  --no-resume
```

### Resume

```bash
python scripts/25_train_transmil_virchow2_binary.py \
  --config configs/transmil_virchow2_train_binary.yaml \
  --device cuda \
  --resume
```

### Evaluacion oficial

```bash
python scripts/26_evaluate_transmil_virchow2_binary.py \
  --config configs/transmil_virchow2_train_binary.yaml \
  --device cuda
```

## Notas

`test_attention_scores.csv` registra atencion auxiliar por tile para apoyar la
revision de regiones relevantes. El orden corresponde al orden de tiles usado
por el `.pt`; si una WSI supera `max_tiles`, TransMIL usa los primeros
`max_tiles`.

No subir a GitHub embeddings, checkpoints, metricas generadas, graficos ni
outputs pesados. Todo eso debe permanecer en Google Drive.
