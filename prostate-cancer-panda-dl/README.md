# Prostate Cancer PANDA DL

## Objetivo del proyecto
Desarrollar un sistema de vision computacional con Deep Learning, guiado por regiones relevantes, para la deteccion y gradacion de severidad del cancer de prostata en biopsias histopatologicas digitalizadas (WSI), con enfoque de investigacion para tesis/articulo cientifico con potencial de publicacion.

## Dataset usado
- Competencia/Dataset: **PANDA - Prostate cANcer graDe Assessment**
- Ruta esperada en Kaggle Notebook:
  - `/kaggle/input/prostate-cancer-grade-assessment/`
- Archivos/carpetas esperados:
  - `train.csv`
  - `train_images/`
  - `train_label_masks/`
  - `test_images/`
  - `sample_submission.csv`

## Alcance de esta primera fase
Esta fase solo valida que el dataset puede leerse correctamente desde Kaggle:
- Verifica rutas y archivos requeridos.
- Carga `train.csv`.
- Reporta columnas, cantidad de registros y distribuciones de `isup_grade` y `gleason_score`.
- Cuenta archivos `.tiff` en imagenes y mascaras.
- Intenta abrir 2-3 WSI y sus mascaras (si existen), mostrando metadata basica.

## Que NO hace todavia
- No entrena modelos.
- No extrae tiles.
- No extrae embeddings.
- No aplica preprocesamiento avanzado.
- No implementa pipeline de inferencia/modelado.

## Estructura inicial
```text
prostate-cancer-panda-dl/
├── README.md
├── requirements.txt
├── config.yaml
├── AGENTS.md
├── notebooks/
│   └── kaggle_validate_panda.ipynb
├── scripts/
│   └── 01_validate_panda_access.py
└── src/
    └── utils/
        ├── paths.py
        └── seed.py
```

## Ejecucion en Kaggle
1. Crear/abrir un notebook de Kaggle.
2. Adjuntar el dataset **prostate-cancer-grade-assessment**.
3. Copiar este repositorio al entorno de Kaggle (por ejemplo en `/kaggle/working/`).
4. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
5. Ejecutar validacion por script:
   ```bash
   python scripts/01_validate_panda_access.py --config config.yaml
   ```
6. Opcional: ejecutar el notebook `notebooks/kaggle_validate_panda.ipynb` para la misma validacion por bloques.
