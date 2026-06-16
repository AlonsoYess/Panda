# Extraccion de embeddings Virchow2 para PANDA

## Objetivo

Esta fase extrae embeddings histopatologicos con **Virchow2** para el proyecto
PANDA de deteccion binaria de cancer de prostata.

La tarea sigue siendo binaria:

- `cancer_label = 0`: no cancer
- `cancer_label = 1`: cancer

Esta fase no entrena modelos MIL. ABMIL + Virchow2 y TransMIL + Virchow2 se
implementaran en fases posteriores.

## Entrada

El extractor usa los mismos ZIPs por batch, manifiestos, tiles seleccionados y
splits `train`, `valid`, `test` usados previamente para UNI2-h.

Ruta esperada en Google Drive:

```text
/content/drive/MyDrive/PANDA_PROSTATE/data/raw_batches
```

Cada ZIP debe contener una estructura compatible con:

```text
batch_XXXX_YYYY/
├── selected_tiles/
├── metadata/
│   ├── tile_manifest.csv
│   └── candidate_tiles_manifest.csv
└── summary.json
```

Tambien se soporta una carpeta raiz adicional como:

```text
panda_outputs_batches/batch_XXXX_YYYY/
```

## Salida

Los embeddings se guardan en:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings
```

Estructura esperada:

```text
outputs/virchow2_binary/
├── embeddings/
│   ├── train/{slide_id}.pt
│   ├── valid/{slide_id}.pt
│   └── test/{slide_id}.pt
├── metrics/
│   └── virchow2_embedding_summary.csv
└── logs/
    └── virchow2_embedding_errors.csv
```

Cada `.pt` conserva trazabilidad compatible con UNI2-h:

- `slide_id`
- `features`
- `tile_ids`
- `tile_paths`
- `coordinates`
- `split`
- `cancer_label`
- `isup_grade`
- `gleason_score`
- `encoder_name`
- `encoder_family`
- `embedding_dim`
- `image_size`
- `transform_info`
- `source_zip`
- `source_manifest_path`
- `manifest_hash`
- `created_at`
- `software_versions`
- `git`
- `cuda`

## Token de Hugging Face

Virchow2 puede requerir acceso aprobado en Hugging Face si el modelo es privado
o gated.

El token debe configurarse como variable de entorno `HF_TOKEN`, por ejemplo desde
Colab Secrets. No debe escribirse en codigo, YAML, notebooks ni documentacion.

## Comandos

### Dry-run

Carga Virchow2, toma una slide, procesa pocos tiles, ejecuta forward y no guarda
ningun `.pt`.

```bash
python scripts/20_extract_virchow2_embeddings.py \
  --config configs/virchow2_extract_binary.yaml \
  --dry-run \
  --device cuda
```

### Smoke test

Procesa 5 WSI de train y fuerza regeneracion si ya existieran.

```bash
python scripts/20_extract_virchow2_embeddings.py \
  --config configs/virchow2_extract_binary.yaml \
  --device cuda \
  --max-slides 5 \
  --splits train \
  --force
```

### Extraccion oficial

```bash
python scripts/20_extract_virchow2_embeddings.py \
  --config configs/virchow2_extract_binary.yaml \
  --device cuda
```

### Resume / continuacion

El resume se hace ejecutando el mismo comando. Como `skip_existing: true`, los
`.pt` validos existentes se verifican y se saltan.

```bash
python scripts/20_extract_virchow2_embeddings.py \
  --config configs/virchow2_extract_binary.yaml \
  --device cuda
```

## Validacion

El extractor valida que cada `.pt` tenga:

- `features` como `torch.Tensor float32`
- shape `[n_tiles, embedding_dim]`
- valores finitos
- metadata minima completa
- longitudes consistentes en `tile_ids`, `tile_paths` y `coordinates`

Si una slide falla, se registra el error y el proceso continua con la siguiente.

## Advertencia

Virchow2 escribe solamente en:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary
```

No debe escribir en outputs de UNI2-h, ABMIL, CLAM ni TransMIL. No subir a GitHub
embeddings, ZIPs, checkpoints, outputs pesados ni tokens.
