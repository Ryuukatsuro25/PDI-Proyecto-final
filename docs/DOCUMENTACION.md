# Documentación técnica

Explica **qué hace cada parte del código** de `proyecto/ProyectoPDI_Completo.ipynb`, cómo se conectan las celdas y qué hacer si algo falla.

## Contenido
1. [Visión general](#1-visión-general)
2. [Recorrido celda por celda](#2-recorrido-celda-por-celda)
3. [Parámetros de configuración](#3-parámetros-de-configuración)
4. [Entrenamiento de la CNN](#4-entrenamiento-de-la-cnn)
5. [Módulos del sistema](#5-módulos-del-sistema)
6. [Fórmulas](#6-fórmulas)
7. [Excepciones](#7-excepciones)
8. [API REST](#8-api-rest)
9. [Interfaz (frontend)](#9-interfaz-frontend)
10. [Archivos que genera](#10-archivos-que-genera)
11. [Solución de problemas](#11-solución-de-problemas)

---

## 1. Visión general

```
Obra ─→ ImageLoader ─→ PreprocesamientoImagen ─→ ClasificadorDeObras (ResNet18) ─→ Grad-CAM
              ├──────→ ExtractorDePaleta (K-Means en CIE Lab)
              └──────→ AnalizarGestosFaciales (rostros dentro de la obra)
Cámara ─→ CapturaDeFrame (cada 400 ms) ─→ AnalizarGestosFaciales (YuNet + FER+)
                              ↓
               ResultadoProceso (reglas, umbrales, excepciones, valencia)
                              ↓
               Integración (Sistema) ─→ JSON ─→ FastAPI ─→ interfaz web
```

El notebook se ejecuta **de arriba hacia abajo** (*Ejecutar todo*). Cada celda usa variables de las anteriores.

---

## 2. Recorrido celda por celda

### Sección 1 · Importaciones y configuración
| Celda | Qué hace | Crea |
|---|---|---|
| Dependencias | Revisa si están FastAPI, PyTorch, OpenCV, ONNX Runtime, scikit-learn…, e instala con `pip` solo lo que falte | — |
| Importaciones | Importa todas las librerías. Detecta Jupyter/Colab y GPU. Debe mostrar `Usando: cuda (Tesla T4)` | `EN_JUPYTER`, `device` |
| `class config` | **Única celda de configuración**: rutas, parámetros de entrenamiento, umbrales y puerto (sección 3). Crea `app_pdi/modelos/` | `config`, `SEED`, `IMG_SIZE`, `BATCH_SIZE`, `CLASES`, `EXTS` |

### Sección 2 · Carga del dataset
| Celda | Qué hace | Crea |
|---|---|---|
| ImageLoader | `cargar_imagen(bytes)`: valida que haya datos, ≤ 10 MB, JPEG/PNG, que se pueda decodificar y ≥ 32 px; corrige la orientación EXIF. `ErrorProceso` es la excepción controlada del proyecto (tipo, mensaje en español, gravedad, módulo) | `cargar_imagen`, `ErrorProceso` |
| Localizar el dataset | Monta Drive. Busca la carpeta del dataset **sin importar mayúsculas, tildes, espacios ni guiones**: primero la ruta de `config`, luego accesos directos a carpetas compartidas, unidades compartidas y una búsqueda automática. Empareja `Negativo/`, `positivo/`, etc. con las clases internas, y localiza `precarga-sin-analizar`. Si no encuentra nada, imprime un **diagnóstico** con las carpetas que hay en Drive | `DATASET`, `CARPETA_CLASE`, `PRECARGA` |
| Inventario y folds | Abre cada imagen, la etiqueta por su carpeta (negativo = 0, positivo = 1) y la guarda a 224×224 en memoria. Detecta duplicados (MD5 exacto y dHash ≤ 4 bits) con *union-find*, calcula la media y la desviación RGB, define el aumento de datos, el `Dataset` de PyTorch y la **validación cruzada `StratifiedGroupKFold` de 5 folds**. Muestra un lote de ejemplo | `muestras`, `N`, `etiquetas`, `grupos`, `duplicados`, `folds`, `FOLD_DE`, `crear_loader` |
| Modelos de rostro | Descarga, si faltan, YuNet (detector de rostros) y FER+ (8 emociones) en formato ONNX | archivos `.onnx` |
| Imágenes precargadas | `DatasetPrecarga`: índice de la carpeta de precarga, imágenes aleatorias, lectura segura por nombre y aviso si la CNN vio una imagen | `DatasetPrecarga` |

### Sección 3 · Entrenamiento y análisis de rostros
| Celda | Qué hace |
|---|---|
| Arquitecturas | `ResNetObras`: ResNet18 de ImageNet con `conv1…layer3` congelados y `layer4` + `Dropout 0.5 → Linear(512, 2)` entrenables. `CNN`: red del Parcial 2 (3 bloques Conv-ReLU-MaxPool + 2 capas densas). Ambas exponen `features()` y `head()` para Grad-CAM |
| Funciones de entrenamiento | `una_epoca` (entrena o evalúa, con AMP en GPU), `validacion_cruzada`, `entrenar_final`, `calcular_metricas`, `resumen_cv`, `tabla_por_imagen`, `graficar_cv` |
| **Entrenamiento** | Validación cruzada de la ResNet18 → elección de la mejor época → modelo final con las 500 imágenes → guarda el modelo, las predicciones, las métricas y la partición → copia a Drive → comparación con la CNN del Parcial 2. Detalle en la sección 4 |
| PreprocesamientoImagen | Imagen → tensor normalizado `[1, 3, 224, 224]`. `medir_calidad` calcula el brillo y la nitidez (varianza del Laplaciano) |
| ClasificadorDeObras + Grad-CAM | Carga `cnn_obras.pt`. `predecir` da los logits; `gradcam` calcula el mapa de calor; `regiones_desde_cam` saca las 1-2 zonas más influyentes |
| ExtractorDePaleta | 6 colores predominantes con K-Means en CIE Lab, con nombre en español, proporción, brillo, saturación y % de colores cálidos |
| AnalizarGestosFaciales | YuNet → recorte del rostro → FER+ (64×64 en gris) → 8 probabilidades. Gestos a partir de los 5 puntos (inclinación, giro, ancho de boca, mirada al frente) y aptitud del rostro 0-100 |
| ResultadoProceso | Reglas de aceptación y excepciones: confianza de la obra ≥ 60 %, rostro ≥ 0.80 y ≥ 6 % del cuadro, emoción ≥ 40 %, brillo entre 45 y 225, nitidez ≥ 25. Promedia la secuencia de fotogramas, calcula la valencia y la coincidencia |
| Integración | `Sistema` reúne todo: `analizar` (obra + fotogramas), `analizar_frame` (vista en vivo), `analizar_dataset` (precargadas). Se crea `sistema = Sistema()` e imprime su estado |

### Sección 4 · Resultados
| Celda | Qué muestra |
|---|---|
| Curvas y métricas | Curvas de pérdida y accuracy (media ± desviación de 5 folds), `classification_report`, matriz de confusión, ROC, tabla por fold y comparación de modelos |
| Imagen por imagen | Aciertos por clase, % con confianza ≥ 60 %, histograma de separación, tabla de predicciones y galerías: errores más seguros, aciertos más seguros, imágenes dudosas |
| Ejemplo | Flujo completo sobre una imagen: rostros detectados, Grad-CAM, paleta, interpretación y tiempos. La imagen del ejemplo pertenece al entrenamiento: la cifra válida es la de la validación cruzada |

### Sección 5 · Aplicativo web
| Celda | Qué hace |
|---|---|
| index.html / styles.css / app.js | Guardan la interfaz en variables (`INDEX_HTML`, `STYLES_CSS`, `APP_JS`). Las mismas están como archivos en `proyecto/frontend/` |
| API | Crea la app FastAPI con 12 rutas (sección 8) |
| Servidor | Inicia uvicorn en un hilo en segundo plano. Si el puerto 8000 ya está en uso (por ejemplo, al volver a ejecutar), avisa y deja el servidor anterior |
| Diagnóstico | Lista los hilos y llama a `/api/estado` desde el propio entorno. Si dice *Conexión exitosa*, el servidor funciona |
| Túnel público | Instala Localtunnel, muestra la IP de Colab (es la contraseña de la página) y crea un enlace `https://….loca.lt`. **Esta celda se queda ejecutándose** mientras se usa la app. Borra su salida antes de compartir el notebook |

---

## 3. Parámetros de configuración

| Parámetro | Valor | Significado |
|---|---|---|
| `DRIVE_PROYECTO` | `/content/drive/MyDrive/Proyecto Final PDI VERSION 3` | Carpeta del proyecto en Drive |
| `DATASET_DRIVE` | `DRIVE_PROYECTO/dataset usado` | Carpeta con `positivo/` y `negativo/` |
| `PRECARGA_DIR` | `…/Precarga sin analizar` | Imágenes no vistas (se acepta también `precarga-sin-analizar`) |
| `DRIVE_RESULTADOS` | `DRIVE_PROYECTO/resultados_cnn` | Copia persistente del modelo |
| `DIR_TRABAJO` / `DIR_MODELOS` | `/content/app_pdi` / `…/modelos` | Salidas de la sesión |
| `REENTRENAR` | `True` | `True` entrena en cada ejecución; `False` reutiliza el modelo local o de Drive |
| `ENTRENAR_COMPARACION` | `True` | Evalúa también la CNN del Parcial 2 (~4.5 min con GPU) |
| `CLASES` | `["negativo", "positivo"]` | Etiquetas **internas**, siempre en minúscula. Las carpetas pueden llevar mayúscula |
| `EPOCAS` / `EPOCAS_CNN` | `10` / `30` | Épocas por fold |
| `LR` / `LR_CABEZA` / `LR_CNN` | `1e-4` / `1e-3` / `1e-3` | Tasas de aprendizaje |
| `DROPOUT` / `WEIGHT_DECAY` | `0.5` / `1e-4` | Regularización |
| `BATCH_SIZE` / `IMG_SIZE` / `K_FOLDS` | `32` / `224` / `5` | Lote, tamaño de entrada, folds |
| `UMBRAL_CONFIANZA_OBRA` | `0.60` | Por debajo: *revisión requerida* |
| `UMBRAL_CONFIANZA_ROSTRO` / `TAM_MIN_ROSTRO` | `0.80` / `0.06` | Aceptación del rostro del visitante |
| `UMBRAL_EMOCION` | `0.40` | Por debajo: expresión ambigua |
| `BRILLO_MIN` / `BRILLO_MAX` / `NITIDEZ_MIN` | `45` / `225` / `25` | Calidad de la imagen |
| `N_COLORES` / `MAX_FRAMES` / `PUERTO` | `6` / `60` / `8000` | Paleta, fotogramas máximos, puerto |

---

## 4. Entrenamiento de la CNN

1. **Validación cruzada.** Para cada uno de los 5 folds se entrena un modelo nuevo con ~400 imágenes y se predicen las ~100 que no vio, en cada época. Al terminar, **cada imagen tiene una predicción de un modelo que no la vio**.
2. **Mejor época.** La de menor pérdida media de validación: la **6**.
3. **Modelo final.** Se entrena con las 500 imágenes durante 6 épocas y se guarda en `cnn_obras.pt` junto con la arquitectura, la normalización y las clases.
4. **Archivos.** `predicciones_por_imagen.csv`, `validacion_cruzada.npz`, `metricas_cnn.json` (lo muestra *Acerca del modelo*) y `particion_cnn.json`. Todo se copia a Drive. Si la carpeta es de solo lectura, avisa sin detenerse.
5. **Comparación.** La CNN del Parcial 2 con los mismos folds (mejor época: 16).

Optimización:
- ResNet18: AdamW (`layer4` a 1e-4 y capa final a 1e-3), CrossEntropy, precisión mixta (AMP) en GPU.
- Aumento de datos: recorte aleatorio de 75-100 %, volteo horizontal, rotación ±8° y ColorJitter suave.

---

## 5. Módulos del sistema

| Módulo | Entrada → salida |
|---|---|
| `cargar_imagen` | bytes → RGB `uint8` (o `ErrorProceso`) |
| `PreprocesamientoImagen` | RGB → tensor `[1, 3, 224, 224]` |
| `ClasificadorDeObras.predecir` / `.gradcam` | tensor → logits `[1, 2]` / mapa 224×224 en 0-1 |
| `ExtractorDePaleta` | RGB → `{colores[6], brillo_medio, saturacion_media, proporcion_calida}` |
| `AnalizarGestosFaciales` | RGB → `{estado, detecciones[]}` con caja, puntos, emociones, gestos y aptitud |
| `ResultadoProceso` | salidas crudas → resultados validados + excepciones |
| `Sistema.analizar` | obra + fotogramas → `{estado, obra, rostro, coincidencia, nivel_coincidencia, interpretacion, excepciones, tiempos_ms}` |

---

## 6. Fórmulas

| Concepto | Fórmula |
|---|---|
| Valencia de la obra | `v_obra = p(positivo) − p(negativo)` ∈ [−1, 1] |
| Valencia de cada emoción | felicidad +1 · sorpresa +0.3 · neutral 0 · desprecio −0.5 · miedo −0.7 · tristeza −0.8 · asco −0.8 · enojo −0.9 |
| Valencia del visitante | `v = Σ p(emoción) · valencia(emoción)`, promedio de los fotogramas |
| **Coincidencia** | `1 − |v_obra − v_visitante| / 2` ∈ [0, 1] · alta ≥ 0.7 · media ≥ 0.4 · baja < 0.4 |
| Aptitud del rostro | 30 % confianza del detector + 20 % tamaño + 20 % frontalidad + 15 % nitidez + 15 % iluminación |
| Grad-CAM | `ReLU(Σ_k α_k A_k)` con `α_k` = media de los gradientes de la clase sobre el canal *k* |

**Limitación conocida:** una cara neutral tiene valencia 0, así que frente a una obra clasificada con seguridad (v ≈ ±1) la coincidencia queda en ~50 %, aunque no haya expresión que comparar.

---

## 7. Excepciones

| `tipo` | Cuándo ocurre |
|---|---|
| `archivo_ausente` / `archivo_corrupto` / `formato` | No llegó archivo, no se pudo leer, o no es JPG/PNG ≤ 10 MB |
| `fuera_de_rango` | Imagen < 32 px, confianza de la CNN < 60 % o emoción < 40 % |
| `calidad_insuficiente` | Imagen muy oscura, sobreexpuesta o borrosa |
| `rostro_no_detectado` | Ningún fotograma con rostro, o muy pocos |
| `multiples_rostros` | Varias personas en cuadro: se usa la más cercana |
| `entrada_incompleta` | Falta la obra o el rostro para calcular la coincidencia |
| `servidor` | CNN o FER+ no disponibles |
| `imagen_conocida` | La imagen precargada formó parte del entrenamiento |
| `dataset_no_encontrado` / `imagen_no_encontrada` | Problemas con la carpeta de precarga |

La GUI muestra el paso *Excepciones* cuando hay errores o avisos.

---

## 8. API REST

| Método y ruta | Descripción |
|---|---|
| `GET /` · `/styles.css` · `/app.js` | Interfaz |
| `GET /api/estado` | Estado de la CNN, del detector y de FER+, y umbrales |
| `GET /api/metricas` | `metricas_cnn.json` |
| `POST /api/frame` | Un fotograma (`frame`) → análisis rápido para la vista en vivo |
| `POST /api/analizar` | `obra`, `frames[]`, `tiempos` (JSON), `modo` → resultado completo |
| `GET /api/dataset/estado` · `POST /api/dataset/recargar` | Índice de imágenes precargadas |
| `GET /api/dataset/aleatorias?n=6` | Imágenes al azar (máximo 24) |
| `GET /api/dataset/imagen?nombre=` | Una imagen precargada |
| `POST /api/dataset/analizar` | Analiza una precargada (`nombre`) |

---

## 9. Interfaz (frontend)

`proyecto/frontend/index.html` (estructura), `styles.css` (diseño) y `app.js` (lógica):

| Bloque de `app.js` | Funciones |
|---|---|
| Entrada | `setObra`, `validarArchivo`, `cargarEstado` |
| Cámara y vista en vivo | `iniciarCamara`, `detenerCamara`, `cicloLive`, `pintarLive` |
| Observación | `observar` (3, 6 o 10 s; 1 fotograma cada 400 ms), `fotoUnica` |
| Análisis | `analizar`, `renderExcepciones` |
| Resultado | `renderResultado`, `renderClasificacion`, `renderPaleta`, `renderEmociones`, `renderGestos`, `renderTiempo`, `gaugeSVG`, `escalaValencia` |
| Precargadas | `pedirAleatorias`, `analizarUna`, `bombear` (2 en paralelo), `abrirDetalle`, `usarConCamara` |
| Utilidades | `fetchRetry` (reintentos y tiempo límite), `toast`, `irAPaso`, `announce` (accesibilidad) |

---

## 10. Archivos que genera

| Ruta | Contenido |
|---|---|
| `app_pdi/modelos/cnn_obras.pt` | Pesos del modelo final + metadatos |
| `app_pdi/modelos/metricas_cnn.json` | Métricas y curvas (para la GUI) |
| `app_pdi/modelos/particion_cnn.json` | Imágenes que vio la CNN |
| `app_pdi/modelos/predicciones_por_imagen.csv` | Predicción de cada imagen en la validación cruzada |
| `app_pdi/modelos/validacion_cruzada.npz` | Probabilidades por época y curvas |
| `app_pdi/modelos/*.onnx` | YuNet y FER+ |
| `Drive/…/resultados_cnn/` | Copia persistente de lo anterior |

---

## 11. Solución de problemas

| Síntoma | Solución |
|---|---|
| `No se encontró el dataset` + diagnóstico | Lee el diagnóstico: muestra qué carpetas hay. Si la carpeta es compartida, agrégale un acceso directo en *Mi unidad* |
| El entrenamiento tarda más de una hora | La sesión no tiene GPU: *Cambiar tipo de entorno → T4 GPU* |
| Quieres evitar reentrenar | `REENTRENAR = False`: usa el modelo de `resultados_cnn/` |
| `El dataset cambió desde el último entrenamiento` | Se agregaron o quitaron imágenes: `REENTRENAR = True` |
| `CUDA out of memory` | `BATCH_SIZE = 16` |
| *El puerto 8000 ya está en uso* | El servidor de una ejecución anterior sigue activo y la app funciona. Para aplicar cambios del código: *Reiniciar sesión* y *Ejecutar todo* |
| La cámara no se activa | Usa el enlace `https://….loca.lt` y permite la cámara en el navegador |
| Localtunnel pide contraseña | Es la IP que muestra la celda del túnel |
| *Rostro no detectado* | Luz frontal, a 50-80 cm de la cámara, de frente |
