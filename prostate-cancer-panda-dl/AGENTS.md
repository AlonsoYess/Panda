# AGENTS - Reglas Operativas del Proyecto PANDA

## Alcance y control de fases
- No avanzar a entrenamiento de modelos sin autorizacion explicita del investigador.
- No implementar extraccion de tiles, embeddings o modelado en esta fase inicial.
- Mantener el enfoque del proyecto: deteccion y gradacion de cancer de prostata con PANDA.

## Reproducibilidad
- Mantener rutas configurables desde `config.yaml`.
- Mantener semilla fija en scripts para muestreo/validaciones reproducibles.
- Evitar decisiones ocultas o cambios no documentados en el flujo.

## Trazabilidad obligatoria para fases futuras
- Conservar trazabilidad por:
  - `slide_id`
  - `tile_id`
  - `coordenadas`
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
