# Crack Detection — Semantic Segmentation (DeepLabV3+)

Pipeline de **detección de grietas** mediante **segmentación semántica binaria** (*crack* vs *background*), optimizado para **grietas finas/pequeñas**.  
Implementación principal en **PyTorch + segmentation-models-pytorch** y notebook reproducible.

---

## 📁 Contenido del repositorio

Este repositorio incluye **solo**:

- `notebooks/Detección_de_grietas.ipynb` (pipeline completo: preparación, entrenamiento, validación e inferencia)
- Checkpoint del modelo (descarga externa)

**No** se incluyen datasets en el repositorio (son pesados). Se proveen por enlace.

---

## 📦 Descargas (modelo y datasets)

### Modelo entrenado (checkpoint ~150 MB)
- Carpeta en Google Drive: https://drive.google.com/drive/folders/1P7U6NUe7esgJlDOMAstq0LxmOlaBcPrs?usp=sharing

**Cómo usarlo**
1. Descarga el archivo del modelo desde la carpeta.
2. Colócalo en tu repo en: `checkpoints/` (ej.: `checkpoints/final_crack_model.pth`)
3. En el notebook, ajusta la ruta del checkpoint si fuera necesario.

### Datasets (ZIP)
- `Dataset.zip`: https://drive.google.com/file/d/1K5XtI6jlOtGc3KOgEmu0662HT2rbG2Fu/view?usp=sharing
- `DeepCrack.zip`: https://drive.google.com/file/d/1pgiC-P5TekXht-Tl93wy3CnpgesevOve/view?usp=sharing

**Extracción (Colab / Linux)**
```bash
unzip -q Dataset.zip -d /content/Dataset_unzipped
unzip -q DeepCrack.zip -d /content/deepcrack_unzipped
```

> En Windows puedes extraer con 7-Zip y apuntar el notebook a las carpetas resultantes.

---


## 📌 Qué hace este repo

- Entrena un modelo de segmentación para detectar grietas en imágenes.
- Une múltiples datasets (RGB–máscara), los normaliza y crea un índice unificado.
- Prioriza el *recall* de grietas pequeñas con:
  - **DeepLabV3+** (mejor detalle fino).
  - **Focal + Dice** (balance entre falsos negativos y estabilidad).
  - **Oversampling** de ejemplos con grietas pequeñas.
  - **Post-procesamiento morfológico** para consolidar trazos finos y filtrar ruido.

---


### Datasets usados
- DeepCrack
- Pavement Crack Datasets: CRACK500, GAPs384, CrackTree200
- CrackForest (CFD)

## 🧠 Enfoque

- **Tipo de tarea:** Segmentación semántica binaria
- **Entrada:** Imagen (en este proyecto se usa **grayscale** por defecto)
- **Salida:** Máscara binaria (0/1) con píxeles de grieta
- **Modelo:** DeepLabV3+ (encoder EfficientNet)
- **Loss:** FocalDiceLoss (Focal + Dice)

---

## ✅ Requisitos

### Opción A: Google Colab (recomendado)
Este proyecto fue trabajado con flujo tipo Colab (Drive + extracción de ZIPs).

### Opción B: Local (Linux/Windows)
- Python 3.9+
- (Opcional) GPU CUDA para entrenar más rápido

---


## ⚡ Quickstart (Colab)

1. Sube el notebook a Colab (o ábrelo desde tu repo).
2. Monta Drive y descarga/ubica:
   - `Dataset.zip`
   - `DeepCrack.zip`
   - el checkpoint del modelo (si solo harás inferencia)
3. Extrae los ZIPs (ver sección **📦 Descargas**).
4. Ejecuta el notebook de arriba hacia abajo.

> Si solo quieres **inferir** (sin entrenar), ejecuta únicamente: imports → carga del checkpoint → bloque de inferencia/visualización.


## 📦 Instalación

### Dependencias mínimas
```bash
pip install -U pip
pip install segmentation-models-pytorch albumentations timm
pip install torchmetrics opencv-python-headless
pip install scikit-image scipy pandas numpy pillow pyyaml matplotlib
```

> Si vas a usar OpenCV con interfaz gráfica local, cambia `opencv-python-headless` por `opencv-python`.

---

## 🗂️ Estructura sugerida del repo

> Si solo tienes el notebook, puedes dejarlo así.  
> Si quieres “estándar proyecto Python”, esta estructura es ideal.

```text
crack-detection-pipeline/
├─ notebooks/
│  └─ Detección_de_grietas.ipynb
└─ README.md
```

---

## 📚 Datasets usados en este repo

Los datasets se descargan mediante los enlaces de la sección **📦 Descargas (modelo y datasets)**.

- **DeepCrack** (RGB + máscara binaria)
- **Pavement Crack Datasets** (paquete/repo), que agrupa subconjuntos:
  - **CRACK500**
  - **GAPs384**
  - **CrackTree200** (`cracktree200`)
- **CrackForest (CFD)** — requiere conversión de máscaras desde `.mat` a imagen (binaria)

---

## 📁 Preparación de datos (flujo notebook) (flujo notebook)

### 1) Coloca los ZIPs en tu Drive (o carpeta local)
Ejemplo típico:
- `DeepCrack.zip`
- `Dataset.zip` (colección con CRACK500/GAPS/CrackForest/etc.)

### 2) Extrae los ZIPs
En Colab, el notebook extrae a rutas tipo:
- `/content/deepcrack_unzipped`
- `/content/Dataset_unzipped`

### 3) CrackForest: conversión `.mat` → `.jpg/.png`
Si tienes CrackForest en `.mat`, conviértelo a máscaras imagen (binarias) antes de indexar.

---

## 🧪 Parámetros principales (los del pipeline)

> Ajusta si cambiaste algo.

| Parámetro | Valor |
|---|---|
| Input size | 512×512 |
| batch_size | 8 |
| epochs | 60 |
| optimizer | Adam |
| lr | 3e-4 |
| weight_decay | 5e-5 |
| scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| early stopping | patience=10 |
| encoder | efficientnet-b3 |
| in_channels | 1 (grayscale) |
| classes | 1 |

---

## 🚀 Entrenamiento

### Si trabajas SOLO con notebook
1. Abre `notebooks/Detección_de_grietas.ipynb`
2. Ejecuta en orden:
   - Montaje de Drive / paths
   - Extracción de ZIPs
   - Indexación `df`
   - Split train/val
   - Entrenamiento
3. Se guardan:
   - **best checkpoint**
   - **final model**
   - curvas y figuras

### Si lo modularizas (opcional)
```bash
python -m src.train --config configs/train.yaml
```

---

## 🔍 Inferencia

En notebook:
- Carga el checkpoint
- Corre inferencia en muestras
- Visualiza `img / gt / pred`

(En versión modularizada)
```bash
python -m src.infer --checkpoint checkpoints/best.pth --input path/to/images --out outputs/preds
```

---

## 🧼 Post-procesamiento (grietas finas)

Se usa morfología + filtrado por componentes conectados para:
- Unir segmentos cortados
- Reducir puntos aislados (falsos positivos)
- Favorecer trazos delgados continuos

Parámetros típicos:
- threshold = 0.3
- min_area = 10 px²
- dilate → close → erode → filtro por área

---

## 📈 Métricas

Durante entrenamiento/validación:
- **IoU**
- **Dice**

Binarización típica:
- `sigmoid(pred) > 0.5`

---

## 📦 Artefactos generados

El pipeline suele guardar:
- `training_curves.png`
- `predictions_sample.png`
- `predictions_by_size.png`
- `validation_comparison.png` (comparación de post-process)

> Recomendación: guarda todo en `outputs/` para mantener orden.

---

## 💾 Checkpoints

Ejemplo típico:
- `checkpoints/best_crack_model_optimized.pth`
- `checkpoints/final_crack_model.pth`

Incluye en el README (si aplica):
- dónde se guardan
- qué checkpoint usar para inferencia

---

## 🧯 Troubleshooting

**1) Error: tamaños diferentes (imagen vs máscara)**  
✅ Solución: en indexación/loader aplica resize consistente y binarización de máscara.

**2) OpenCV falla en Colab**  
✅ Usa `opencv-python-headless`.

**3) OOM (memoria GPU)**  
✅ Baja `batch_size`, reduce encoder, baja resolución (512→384).

**4) Predicciones “muy gruesas”**  
✅ Ajusta post-procesamiento (kernel, iteraciones) y/o threshold.

---
---

## 🙌 Créditos

Notebook base: `Detección_de_grietas.ipynb`  
Autor: *Espinoza Herrera Gustavo Diego*  
