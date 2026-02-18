# Detección de grietas (segmentación semántica binaria)

Este documento describe el pipeline implementado en el notebook **Detección_de_grietas (2).ipynb** para **detección de grietas** mediante **segmentación semántica binaria** (grieta vs. no-grieta), con foco en **grietas pequeñas/finas**.

---

## Objetivo

- Entrenar un modelo de segmentación que detecte grietas finas en imágenes de inspección.
- Unificar múltiples datasets (pares RGB–máscara), estandarizar formato y **mejorar recall** en grietas pequeñas mediante:
  - *Loss* optimizado (Focal + Dice).
  - Arquitectura **DeepLabV3+** (mejor preservación de detalle fino).
  - Balanceo por sobremuestreo de ejemplos con grietas pequeñas.
  - Post-processing morfológico para consolidar discontinuidades y filtrar ruido.

---

## Stack y dependencias (Colab)

El notebook está pensado para **Google Colab** (usa `!pip install` y `drive.mount`).

```bash
pip install segmentation-models-pytorch albumentations timm
pip install torchmetrics opencv-python-headless
pip install scikit-image scipy
```

> Si lo migras a ejecución local, cambia los paths de Drive y elimina los comandos con `!`.

---

## Datasets y estructura esperada

El pipeline asume que tienes **dos ZIPs** en tu Drive (misma carpeta):

- `DeepCrack .zip`
- `Dataset.zip` (dataset adicional con subcarpetas tipo CRACK500 / GAPS384 / cracktree200 / CrackForest)

En Colab se extraen en:

- `/content/deepcrack_unzipped`
- `/content/Dataset_unzipped`

### Conversión CrackForest (.mat → .jpg)

Las máscaras de CrackForest (en `.mat`) se convierten a imágenes **.jpg** (binarias / escala de grises) en una carpeta dedicada:

- Entrada: `.../CrackForest-dataset-master/black_gray`
- Salida:  `.../CrackForest-dataset-master/black_gray_jpg`

---

## Indexación de pares RGB–máscara

Se construye un DataFrame (`df`) con columnas típicas:

- `id`
- `dataset`
- `rgb_path`
- `mask_path`

Luego se hace limpieza de casos sin RGB o rutas inválidas.

---

## Análisis de tamaño de grieta

Para cada máscara:
- Se calcula `crack_ratio = crack_pixels / total_pixels`.
- Se obtiene número y tamaño de componentes conectados (`connectedComponentsWithStats`).

Clasificación por bins (según `crack_ratio`):

- `very_small`: 0 – 0.005
- `small`: 0.005 – 0.02
- `medium`: 0.02 – 0.05
- `large`: 0.05 – 1.0

---

## Balanceo del dataset (grietas pequeñas)

Se sobremuestran (oversampling) muestras cuyo `size_category` ∈ {`very_small`, `small`}.

- `oversample_factor = 2`
- Muestra con reemplazo (`replace=True`)
- `random_state = 42` para reproducibilidad

---

## Preprocesamiento y augmentations

### Input
- Lectura en **escala de grises** (`cv2.IMREAD_GRAYSCALE`)
- `in_channels = 1` en el modelo
- Las máscaras se binarizan (`mask > 0`) y se **alinean** (rotación/transposición/resize) si la forma no coincide con la imagen.

### Augmentations (train)
- Resize: **512×512**
- Flips: horizontal/vertical
- Rotaciones: 90° y rotación con límite
- Iluminación: brightness/contrast, gamma
- Ruido/blur: GaussNoise o GaussianBlur (probabilístico)
- Normalización: mean=0.449, std=0.226

### Val
- Resize + Normalización (sin augmentations geométricos)

---

## Modelo

Arquitectura:
- **DeepLabV3+** (segmentation_models_pytorch)
- Encoder: **efficientnet-b3**
- `in_channels=1`, `classes=1`, `activation=None` (se usa sigmoid para inferencia)

---

## Loss y optimización

Loss principal:
- **FocalDiceLoss**
  - alpha=0.8
  - gamma=2.0
  - dice_weight=1.0
  - focal_weight=0.5

Optimizer:
- Adam
- LR inicial: **3e-4**
- weight_decay: **5e-5**

Scheduler:
- ReduceLROnPlateau (mode=min, factor=0.5)
- patience=5

Early stopping:
- patience=10

---

## Métricas

Durante train/val se calculan:

- **IoU** (binarizando `sigmoid(pred) > 0.5`)
- **Dice** (binarizando `sigmoid(pred) > 0.5`)

---

## Entrenamiento

Parámetros principales:

- epochs: **60**
- batch_size: **8**
- split train/val: 80/20 (muestreo aleatorio, `random_state=42`)

Se guarda el **mejor checkpoint** al mejorar Dice de validación:

- `/content/best_crack_model_optimized.pth`

Además se guarda un paquete final (modelo + optimizador + history):

- `/content/final_crack_model.pth`

---

## Post-processing (grietas pequeñas)

Se aplica post-procesamiento morfológico para consolidar trazos finos y filtrar falsos positivos pequeños:

- Threshold (probabilidad): **0.3**
- min_area: **10** px²
- Dilate (3×3, 2 iteraciones) → Close (5×5) → Erode (2×2) → filtrado por componentes conectados

Función clave: `enhance_small_cracks(pred_mask, min_area=10, threshold=0.3)`

---

## Inferencia y visualización

Se generan figuras y artefactos típicos en `/content/`:

- `training_curves.png`
- `predictions_sample.png`
- `predictions_by_size.png`

También hay un bloque que busca ejemplos de **grietas muy pequeñas** filtrando por `mask_ratio` aproximado (ej. 0.001–0.005).

---

## Validación avanzada (comparación de post-processings)

El notebook incluye una validación avanzada que compara 4 estrategias:

- `enhance_small_cracks` (original)
- `enhance_thin_cracks_v1` (threshold bajo + morfología)
- `enhance_thin_cracks_v2` (adaptive threshold + morfología)
- `enhance_thin_cracks_v3` (multi-threshold + morfología)

Se calcula una métrica rápida de **cobertura (%)** basada en proporción de píxeles positivos, y se guarda un comparativo:

- `validation_comparison.png`

Además, existe un flujo `quick_validation()` para subir una imagen (Colab) y ejecutar la comparación automáticamente.

---

## Convenciones recomendadas (estándar “proyecto Python”)

Si vas a convertir esto en repositorio mantenible, lo típico es:

- Formato: **PEP8** + `black`
- Estructura sugerida:
  ```text
  crack-detection/
  ├─ notebooks/
  │  └─ Detección_de_grietas.ipynb
  ├─ src/
  │  ├─ data/           # indexación, conversiones, splits
  │  ├─ models/         # definición de modelo
  │  ├─ losses/         # FocalDice, Tversky, etc.
  │  ├─ train.py        # entrenamiento
  │  ├─ infer.py        # inferencia
  │  └─ postprocess.py  # morfología y filtros
  ├─ requirements.txt
  └─ README.md
  ```

---

## Notas rápidas

- **Grayscale**: si migras a RGB, cambia `in_channels=3`, lectura con `cv2.IMREAD_COLOR` y la normalización.
- **Alineación de máscaras**: el notebook incluye lógica para transponer/rotar/rescalar máscaras cuando no coinciden con la imagen.
- **Reproducibilidad**: hay `random_state=42` en varias partes, pero el pipeline completo (PyTorch) puede requerir fijar seeds adicionales si deseas reproducibilidad estricta.

---

## Créditos

Notebook original: **Detección_de_grietas (2).ipynb**  
Documento generado automáticamente como guía de implementación y reproducibilidad.
