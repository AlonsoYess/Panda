# Prostate Cancer PANDA DL

## Objetivo del proyecto
Disenar un sistema de vision computacional con Deep Learning, guiado por regiones relevantes, para deteccion y gradacion de severidad del cancer de prostata en biopsias histopatologicas digitalizadas (WSI), con enfoque de tesis/articulo cientifico.

## Dataset
- Dataset: **PANDA - Prostate cANcer graDe Assessment**
- Ruta de Kaggle usada en este proyecto:
  - `/kaggle/input/competitions/prostate-cancer-grade-assessment`
- Archivos esperados:
  - `train.csv`
  - `train_images/`
  - `train_label_masks/`
  - `sample_submission.csv`

## Fase 1: Validacion de acceso
Valida acceso a rutas, lectura de `train.csv`, apertura de WSI y apertura de mascaras.

No realiza:
- entrenamiento
- extraccion de tiles para modelado
- embeddings
- modelos MIL

Script:
- `python scripts/01_validate_panda_access.py --config config.yaml`

Notebook:
- `notebooks/kaggle_validate_panda.ipynb`

## Fase 2: Preprocesamiento inicial
Prepara artefactos de datos para entrenamiento futuro, sin entrenar modelos.

### Que hace
- Crea `splits.csv` con estratificacion por `isup_grade`.
- Agrega `cancer_label`:
  - `0` si `isup_grade == 0`
  - `1` si `isup_grade >= 1`
- Extrae tiles de `256x256` por grilla usando `OpenSlide.read_region`.
- Evalua candidatos por `tissue_pct` y `mask_pct`.
- Selecciona hasta `tiles_per_slide` por slide (modo controlado con `max_slides`).
- Guarda:
  - `candidate_tiles_manifest.csv`
  - `tile_manifest.csv`
  - PNG de tiles seleccionados
  - logs

### Que NO hace en Fase 2
- No entrena modelos.
- No genera embeddings (UNI/Virchow2).
- No implementa ABMIL/CLAM/TransMIL.
- No calcula metricas finales.

### Outputs de Fase 2
Se guardan en:
- `/kaggle/working/panda_outputs`

Estructura esperada:
```text
/kaggle/working/panda_outputs/
|-- metadata/
|   |-- splits.csv
|   |-- candidate_tiles_manifest.csv
|   `-- tile_manifest.csv
|-- selected_tiles/
|   |-- train/
|   |-- valid/
|   `-- test/
`-- logs/
```

El dataset PANDA original se mantiene en Kaggle (`/kaggle/input/...`).
Luego puedes copiar/exportar la carpeta `panda_outputs` a Google Drive.

## Ejecucion en Kaggle
Dentro del repo:
```bash
pip install -r requirements.txt
python scripts/02_create_splits.py
python scripts/03_extract_tiles.py
```

## Parametros clave
En `config.yaml` puedes ajustar:
- `max_slides` para pruebas controladas
- `tile_size`, `tile_level`, `tiles_per_slide`
- umbrales `min_tissue_pct`, `min_mask_pct`
- proporciones en `split.train_size`, `split.valid_size`, `split.test_size`

## Fase 2C - Extraccion por lotes en Kaggle
Permite procesar PANDA por rangos de slides para evitar ejecutar las 10,616 WSI en una sola corrida.

### Scripts
- `scripts/04_extract_tiles_batch.py`
- `scripts/05_merge_batch_manifests.py`

### Ejemplos de uso
```bash
python scripts/04_extract_tiles_batch.py --batch-index 0 --batch-size 100
python scripts/04_extract_tiles_batch.py --batch-index 1 --batch-size 100
python scripts/04_extract_tiles_batch.py --batch-index 0 --batch-size 100 --split train
python scripts/05_merge_batch_manifests.py
```

### Salida por batch
Cada batch se guarda separado en:
- `/kaggle/working/panda_outputs_batches/batch_XXXX_YYYY/`

Con estructura:
- `metadata/candidate_tiles_manifest.csv`
- `metadata/tile_manifest.csv`
- `logs/04_extract_tiles_batch_YYYYMMDD_HHMMSS.log`
- `selected_tiles/{split}/{slide_id}/{tile_id}.png`
- `summary.json`

### Merge final de manifests
`05_merge_batch_manifests.py` une todos los batches encontrados en:
- `/kaggle/working/panda_outputs_merged/metadata/candidate_tiles_manifest.csv`
- `/kaggle/working/panda_outputs_merged/metadata/tile_manifest.csv`
- `/kaggle/working/panda_outputs_merged/metadata/summary_batches.csv`

## Entrenamiento preliminar ABMIL + UNI en local
Esta fase permite entrenar un clasificador binario (cancer/no cancer) a nivel WSI con:
- encoder congelado `UNI` (solo extraccion de embeddings)
- modelo entrenable `ABMIL`

### Por que UNI y ABMIL
- `UNI` es un encoder fundacional para patologia digital que transforma tiles en vectores de caracteristicas robustos.
- `ABMIL` agrega embeddings de tiles con un mecanismo de atencion para producir una prediccion por WSI.
- El entrenamiento es a nivel WSI porque la etiqueta `cancer_label` es de slide.
- Esta tarea binaria es el primer paso antes de tareas mas complejas (ISUP/Gleason).
- Los resultados en esta etapa son preliminares para validacion metodologica con el profesor.

### Flujo local (Windows)
1. Colocar ZIPs descargados de Kaggle en:
   - `data/raw_batches/`
2. Preparar dataset local:
   ```bash
   python scripts/06_prepare_abmil_dataset.py --zip_dir data/raw_batches --output_dir data/extracted_batches
   ```
3. Configurar token de Hugging Face (`HF_TOKEN`) para acceso a UNI:
   ```powershell
   setx HF_TOKEN "tu_token_aqui"
   ```
   Cierra y vuelve a abrir terminal despues de `setx`.
4. Extraer embeddings UNI:
   ```bash
   python scripts/07_extract_uni_embeddings.py --config configs/abmil_uni_binary.yaml
   ```
5. Entrenar ABMIL binario:
   ```bash
   python scripts/08_train_abmil_binary.py --config configs/abmil_uni_binary.yaml
   ```
6. Evaluar en test:
   ```bash
   python scripts/09_evaluate_abmil_binary.py --config configs/abmil_uni_binary.yaml
   ```

### Nota sobre escalamiento
Esta fase esta pensada para pruebas locales. Luego se puede migrar la misma estructura a Drive/Colab para escalar volumen y tiempo de corrida.

