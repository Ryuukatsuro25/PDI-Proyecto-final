# Arquitectura del sistema

```mermaid
flowchart LR
    subgraph ENTRADA["Entrada"]
        U1["Obra<br/>(subida o precargada)"]
        U2["Cámara autorizada<br/>(consentimiento)"]
    end

    subgraph GUI["Frontend · HTML/CSS/JS"]
        G1["Paso 1 · Entrada"]
        G2["Paso 2 · Análisis"]
        G3["Paso 3 · Excepciones"]
        G4["Paso 4 · Resultado"]
        G5["Formulario de<br/>evaluación"]
        G6["Pestaña<br/>imágenes precargadas"]
    end

    subgraph API["Backend · FastAPI"]
        A1["POST /api/analizar"]
        A2["POST /api/frame<br/>(vista en vivo)"]
        A3["POST /api/evaluacion"]
        A4["GET /api/dataset/*"]
    end

    subgraph OBRA["Rama de la obra"]
        M1["ImageLoader<br/>valida JPG/PNG"]
        M2["PreprocesamientoImagen<br/>224×224 · normalización"]
        M3["ClasificadorDeObras<br/>ResNet18 transfer learning"]
        M4["Grad-CAM<br/>regiones relevantes"]
        M5["ExtractorDePaleta<br/>K-Means en CIE Lab"]
    end

    subgraph ROSTRO["Rama del visitante"]
        F1["CapturaDeFrame<br/>secuencia 6 s · 400 ms"]
        F2["AnalizarGestosFaciales<br/>YuNet + FER+ (8 emociones)"]
    end

    subgraph REGLAS["Reglas e integración"]
        R1["ResultadoProceso<br/>umbrales · calidad · valencia"]
        R2["Coincidencia v3<br/>1 − abs(v_obra − v_visitante) / 2"]
        R3["Integración<br/>JSON para la GUI"]
    end

    subgraph DATOS["Almacenamiento (Google Drive)"]
        D1[("dataset usado/<br/>positivo · negativo")]
        D2[("precarga-sin-analizar")]
        D3[("resultados_cnn/<br/>cnn_obras.pt")]
        D4[("evaluacion/<br/>registros_visitantes.jsonl")]
    end

    U1 --> G1
    U2 --> G1
    G1 --> G2 --> A1
    G1 -. en vivo .-> A2
    G6 --> A4 --> D2
    A1 --> M1
    M1 --> M2 --> M3 --> M4
    M1 --> M5
    A1 --> F1 --> F2
    A2 --> F2
    M3 --> R1
    M5 --> R1
    F2 --> R1
    R1 --> R2 --> R3
    R3 --> G3 --> G4 --> G5 --> A3 --> D4
    D1 -. entrenamiento .-> M3
    D3 -. pesos .-> M3
```

## Flujo de entrenamiento y evaluación

```mermaid
flowchart TB
    A[("500 imágenes<br/>positivo / negativo")] --> B["Inventario · validación<br/>MD5 + dHash (duplicados)"]
    B --> C["StratifiedGroupKFold<br/>5 folds"]
    C --> D["ResNet18 ImageNet<br/>ajuste de layer4 + capa final"]
    C --> E["CNN Parcial 2<br/>(comparación)"]
    C --> F["Regresión logística<br/>rasgos de color (línea base)"]
    D --> G["Predicción fuera de fold<br/>de cada imagen"]
    E --> G
    F --> G
    G --> H["Métricas: accuracy, F1, AUC,<br/>matriz de confusión, ROC"]
    D --> I["Modelo final con las 500<br/>(n.º de épocas = mejor época)"]
    I --> J["Aplicativo"]
    J --> K["Prueba con visitantes reales"]
    K --> L["Métricas del sistema:<br/>detección, FER+ vs. autorreporte,<br/>CNN vs. humanos, coincidencia, SUS"]
```

## Módulos

| Módulo | Entrada | Salida | Técnica |
|---|---|---|---|
| ImageLoader | bytes JPG/PNG (≤ 10 MB, > 32 px) | matriz RGB | Pillow + corrección EXIF; errores con código controlado |
| PreprocesamientoImagen | RGB de tamaño variable | tensor `[1, 3, 224, 224]` | redimensionado + normalización de ImageNet |
| ClasificadorDeObras | tensor | logits de 2 clases | ResNet18 preentrenada; se ajustan `layer4` y la capa final (Dropout 0.5, Weight Decay 1e-4) |
| Grad-CAM | tensor + clase | mapa de calor y regiones | gradientes de `layer4` |
| ExtractorDePaleta | RGB | 6 colores HEX con su proporción | K-Means en CIE Lab |
| CapturaDeFrame | cámara del navegador | secuencia de fotogramas con marca de tiempo | getUserMedia, 1 fotograma cada 400 ms |
| AnalizarGestosFaciales | fotograma RGB | cajas, 5 puntos, 8 emociones, gestos y aptitud del rostro | YuNet (OpenCV Zoo) + FER+ (ONNX Model Zoo) |
| ResultadoProceso | logits, paleta y detecciones | resultados validados + excepciones | softmax, umbrales, brillo y nitidez, valencia |
| Integración | todo lo anterior | JSON para la GUI | coincidencia v3 e interpretación en texto |
| RegistroEvaluacion | resultado + autorreporte | línea JSONL sin imágenes | evaluación con visitantes |
