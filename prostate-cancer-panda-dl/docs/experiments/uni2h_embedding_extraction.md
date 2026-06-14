# Extraccion oficial de embeddings UNI2-h

## Proposito

Esta fase genera representaciones congeladas de tiles histopatologicos PANDA con
`MahmoodLab/UNI2-h`. Los embeddings se usaran posteriormente como entrada para
ABMIL en la tarea binaria `cancer_label`:

- `0`: no cancer
- `1`: cancer

Esta fase no entrena ABMIL ni ajusta UNI2-h.

## UNI clasico y UNI2-h

El baseline historico usa UNI clasico y produce embeddings de 1024 dimensiones.
El experimento oficial usa UNI2-h y exige 1536 dimensiones. Los archivos,
checkpoints y metricas no son intercambiables.

Los outputs se separan para conservar el baseline y evitar que ABMIL reciba
features de encoders incompatibles:

```text
outputs/abmil_uni_binary/       # baseline UNI clasico, 1024-D
outputs/abmil_uni2h_binary/     # experimento UNI2-h, 1536-D
```

## Estructura de Drive

Los ZIPs deben encontrarse en:

```text
/content/drive/MyDrive/PANDA_PROSTATE/data/raw_batches
```

Cada ZIP puede contener directamente `batch_XXXX_YYYY/` o incluir la raiz
adicional `panda_outputs_batches/batch_XXXX_YYYY/`.

Los resultados se escriben en:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary
```

La extraccion temporal se realiza en `/content/panda_uni2h_work`; los tiles no
se descomprimen permanentemente en Drive.

## Preparar Colab

### 1. Configurar el secreto

En Colab, abre el panel **Secrets**, crea una clave llamada `HF_TOKEN`, habilita
su acceso para el notebook y no imprimas su valor. La cuenta de Hugging Face
debe tener acceso aprobado a `MahmoodLab/UNI2-h`.

Al iniciar la sesion, pasa el secreto al entorno sin escribir su valor literal:

```python
from google.colab import userdata
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

### 2. Montar Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

### 3. Clonar el codigo

```bash
%cd /content
!git clone <REPOSITORY_URL>
%cd /content/Panda/prostate-cancer-panda-dl
```

Ajusta el nombre de la carpeta al nombre real del repositorio clonado.

### 4. Instalar dependencias

```bash
!pip install -r requirements.txt
```

## Validacion sin modelo

El dry-run inspecciona ZIPs, estructuras y manifests. No carga UNI2-h, no
descarga pesos y no crea embeddings.

```bash
python scripts/10_extract_uni2h_embeddings.py \
  --config configs/abmil_uni2h_binary.yaml \
  --dry-run \
  --max-wsi 5
```

## Prueba real con 5 WSI

Esta ejecucion carga UNI2-h una sola vez y crea hasta cinco archivos `.pt`
validos:

```bash
python scripts/10_extract_uni2h_embeddings.py \
  --config configs/abmil_uni2h_binary.yaml \
  --max-wsi 5
```

Con una GPU de memoria limitada se recomienda mantener
`batch_size_tiles: 1`. Para sobrescribir embeddings UNI2-h ya validos, agrega
`--force`.

## Extraccion completa

```bash
python scripts/10_extract_uni2h_embeddings.py \
  --config configs/abmil_uni2h_binary.yaml
```

El proceso es reanudable por WSI. Un archivo existente solo se salta despues de
validar que su metadata corresponde a UNI2-h y que sus features son 1536-D.

## Verificar un embedding

```python
from pathlib import Path
import torch

path = next(
    Path(
        "/content/drive/MyDrive/PANDA_PROSTATE/outputs/"
        "abmil_uni2h_binary/embeddings"
    ).rglob("*.pt")
)
payload = torch.load(path, map_location="cpu", weights_only=False)

assert payload["encoder_name"] == "MahmoodLab/UNI2-h"
assert payload["encoder_family"] == "UNI2-h"
assert payload["embedding_dim"] == 1536
assert payload["features"].ndim == 2
assert payload["features"].shape[1] == 1536
```

Una dimension de 1024 corresponde al baseline UNI clasico y el nuevo pipeline
la rechaza explicitamente.

## Reproducibilidad

Cada WSI registra el hash SHA-256 del manifest, ZIP de origen, rutas internas de
tiles, coordenadas, etiquetas disponibles, transformacion, versiones de
software, informacion CUDA/GPU y estado Git.

No subas a Git ZIPs, tiles, embeddings, checkpoints ni resultados pesados.
GitHub contiene codigo y documentacion; Drive contiene los artefactos de datos.
