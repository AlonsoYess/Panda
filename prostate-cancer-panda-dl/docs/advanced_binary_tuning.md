# Tuning avanzado binario basado en validacion

Esta fase agrega un ajuste formal de hiperparametros para los experimentos de
clasificacion binaria cancer/no cancer con embeddings avanzados de 128 tiles por
WSI.

## Principio metodologico

El tuning se realiza exclusivamente con el split de validacion. El split de test
no debe usarse para elegir hiperparametros, arquitectura, threshold ni variantes
de entrenamiento. Test se reserva para una unica evaluacion final despues de
seleccionar las configuraciones con validacion.

## Candidatos iniciales

Se generan variantes solo para cuatro candidatos con encoder Virchow2:

- `Virchow2 + DTFD-MIL`
- `Virchow2 + ABMIL`
- `Virchow2 + CLAM`
- `Virchow2 + ACMIL`

La seleccion parte de estos modelos porque Virchow2 fue el extractor mas
consistente, DTFD-MIL obtuvo alto AUC, ABMIL mantuvo buen balance general, CLAM
mostro alto recall y ACMIL sostuvo sensibilidad competitiva.

## Archivos generados

El generador crea configs en:

```text
configs/tuning_advanced_binary/
```

Tambien crea:

```text
configs/tuning_advanced_binary/tuning_manifest.csv
```

Cada variante escribe en un directorio separado:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/tuning_advanced_binary/<experiment_name>/<variant_name>
```

Por defecto, los resultados de tuning se guardan en Google Drive:

```text
/content/drive/MyDrive/PANDA_PROSTATE/outputs/tuning_advanced_binary
```

Esto evita sobrescribir resultados base existentes y evita que Colab guarde
checkpoints dentro del repositorio temporal.

## Flujo recomendado

1. Generar configs de tuning:

```bash
python scripts/57_generate_tuning_configs_advanced_binary.py
```

Opcionalmente se puede cambiar el destino de resultados:

```bash
python scripts/57_generate_tuning_configs_advanced_binary.py \
  --outputs-root /content/drive/MyDrive/PANDA_PROSTATE/outputs/tuning_advanced_binary
```

2. Entrenar solo las variantes necesarias usando el script del modelo
correspondiente y la config generada. Ejemplo:

```bash
python scripts/44_train_abmil_advanced_binary.py \
  --config configs/tuning_advanced_binary/virchow2_abmil_abmil_v01.yaml
```

3. Recolectar resultados de validacion:

```bash
python scripts/58_collect_tuning_results_advanced_binary.py
```

4. Seleccionar mejores configuraciones:

```bash
python scripts/59_select_best_tuning_configs_advanced_binary.py
```

5. Solo despues de seleccionar por validacion, evaluar test con:

```bash
python scripts/56_evaluate_mil_advanced_binary.py
```

## Criterio de seleccion

La metrica primaria es:

```text
valid_auc
```

El selector tambien marca una alternativa orientada a sensibilidad cuando existe
`valid_recall`. Esto permite reportar dos perspectivas sin contaminar la
seleccion con test:

- mejor AUC de validacion;
- mejor sensibilidad de validacion.

## Nota sobre loss_function

Las configs registran `loss_function` como parte del espacio de tuning:

- `bce`
- `weighted_bce`
- `focal`

Los entrenadores actuales deben interpretar esa clave para que la variante sea
efectiva. Si un entrenador todavia no implementa `focal`, la config conserva la
trazabilidad de la intencion experimental, pero se debe confirmar soporte antes
de ejecutar esa variante como resultado final.

## Regla de tesis

No seleccionar hiperparametros mirando test. Cualquier tabla de test debe
generarse despues de congelar la seleccion usando validacion.
