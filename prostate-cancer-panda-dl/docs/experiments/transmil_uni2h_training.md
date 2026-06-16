# TransMIL + UNI2-h para deteccion binaria

## Objetivo

Este experimento implementa el tercer modelo oficial del proyecto:
**TransMIL + UNI2-h** para clasificacion binaria cancer / no cancer en biopsias
histopatologicas del dataset PANDA.

La etiqueta usada es `cancer_label`:

- `0`: no cancer
- `1`: cancer

No se implementa gradacion ISUP en esta fase.

## Entrada

El entrenamiento usa embeddings UNI2-h previamente generados. No vuelve a leer
imagenes WSI, no lee ZIPs y no procesa tiles PNG.

Ruta esperada de embeddings en Google Drive:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/embeddings
```

Estructura esperada:

```text
embeddings/
├── train/*.pt
├── valid/*.pt
└── test/*.pt
```

Cada archivo `.pt` debe contener embeddings UNI2-h de dimension `1536`.

## Salida

TransMIL escribe solamente en:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/transmil_uni2h_binary
```

Outputs principales:

- `checkpoints/best_model.pt`
- `checkpoints/last_checkpoint.pt`
- `checkpoints/checkpoint_epoch_XX.pt`
- `metrics/train_history.csv`
- `metrics/run_metadata.json`
- `metrics/valid_metrics.json`
- `metrics/test_metrics.json`
- `metrics/best_threshold.json`
- `metrics/valid_predictions.csv`
- `metrics/test_predictions.csv`
- `plots/training_loss.png`
- `plots/validation_auc_f1.png`
- `plots/roc_valid.png`
- `plots/roc_test.png`
- `plots/confusion_matrix_valid.png`
- `plots/confusion_matrix_test.png`

## Diferencia entre ABMIL, CLAM y TransMIL

ABMIL usa atencion para agregar tiles y formar una representacion WSI.

CLAM agrega supervision instance-level sobre tiles de alta y baja atencion.

TransMIL usa un Transformer para modelar relaciones entre tiles y un token CLS
como representacion global de la WSI. Ademas devuelve una atencion auxiliar por
tile para ordenar regiones relevantes.

## Lazy loading

El pipeline usa `validate_on_init=False` al construir el dataset. Esto evita
cargar todos los `.pt` desde Drive durante la inicializacion.

El dataset:

- lista rutas `.pt`
- calcula `len()` desde la cantidad de archivos
- carga cada WSI solo en `__getitem__`

Esto permite que el `dry-run` completo cargue solo un sample y termine rapido.

## Comandos Colab

### Dry-run

```bash
python scripts/15_train_transmil_uni2h_binary.py \
  --config configs/transmil_uni2h_train_binary.yaml \
  --dry-run \
  --device cuda
```

### Smoke test

```bash
python scripts/15_train_transmil_uni2h_binary.py \
  --config configs/transmil_uni2h_train_binary.yaml \
  --epochs 2 \
  --max-train 64 \
  --max-valid 32 \
  --device cuda \
  --no-resume
```

### Entrenamiento oficial

```bash
python scripts/15_train_transmil_uni2h_binary.py \
  --config configs/transmil_uni2h_train_binary.yaml \
  --device cuda \
  --no-resume
```

### Resume si Colab se corta

```bash
python scripts/15_train_transmil_uni2h_binary.py \
  --config configs/transmil_uni2h_train_binary.yaml \
  --device cuda \
  --resume
```

### Evaluacion oficial

```bash
python scripts/16_evaluate_transmil_uni2h_binary.py \
  --config configs/transmil_uni2h_train_binary.yaml \
  --device cuda
```

## Regiones relevantes

El modelo devuelve:

- `logit`: prediccion binaria a nivel WSI
- `attention`: importancia por tile con shape `[n_tiles]`

La atencion auxiliar se calcula solo sobre tokens de tiles, no sobre el token
CLS. El orden de `attention` corresponde al orden de `features` dentro del `.pt`.

Si una WSI contiene mas tiles que `max_tiles`, el modelo trunca a los primeros
`max_tiles`. Con la configuracion actual `max_tiles=512`, esto cubre ampliamente
los experimentos actuales con alrededor de 32 tiles por WSI.

## Advertencia

No subir a GitHub embeddings, checkpoints, metricas pesadas, graficos generados
ni outputs de Drive. El repositorio debe contener solo codigo, configuracion y
documentacion.
