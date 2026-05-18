# AGENTS - Reglas Operativas del Proyecto PANDA

## Alcance y control de fases
- No avanzar a entrenamiento de modelos sin autorizacion explicita del investigador.
- No entrenar modelos sin autorizacion.
- No generar embeddings sin autorizacion.
- Mantener el enfoque del proyecto: deteccion y gradacion de cancer de prostata con PANDA.
- No procesar todo PANDA por defecto en fases de preprocesamiento inicial.
- Usar siempre `max_slides` para pruebas iniciales controladas.

## Reproducibilidad
- Mantener rutas configurables desde `config.yaml`.
- Mantener semilla fija en scripts para muestreo/validaciones reproducibles.
- Evitar decisiones ocultas o cambios no documentados en el flujo.

## Trazabilidad obligatoria para fases futuras
- Conservar trazabilidad por:
  - `slide_id`
  - `tile_id`
  - `coordenadas x/y`
  - `level`
  - `tissue_pct`
  - `mask_pct`
  - `isup_grade`
  - `gleason_score`
  - `split` (train/val/test)
  - `encoder`
  - `modelo`
- Toda nueva fase debe preservar estas llaves para auditoria cientifica.

## Calidad de codigo y comunicacion
- Usar mensajes claros por consola y manejo de errores robusto (`try/except`).
- Documentar supuestos tecnicos y limitaciones de librerias (por ejemplo lectores WSI).
- Priorizar claridad metodologica y compatibilidad con ejecucion en Kaggle.
