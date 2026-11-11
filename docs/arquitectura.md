# Arquitectura del sistema

```mermaid
flowchart LR
    subgraph ENTRADA["Entrada"]
        U1["Obra<br/>(subida o precargada)"]
        U2["Cámara autorizada<br/>(consentimiento)"]
    end
    subgraph GUI["Frontend · HTML/CSS/JS"]
        G1["Paso 1 · Entrada"] --> G2["Paso 2 · Análisis"] --> G3["Paso 3 · Excepciones"] --> G4["Paso 4 · Resultado"]
        G6["Pestaña imágenes precargadas"]
    end
    subgraph API["Backend · FastAPI"]
        A1["POST /api/analizar"]
        A2["POST /api/frame"]
        A4["/api/dataset/*"]
    end
    subgraph OBRA["Rama de la obra"]
        M1["ImageLoader"] --> M2["PreprocesamientoImagen<br/>224×224"] --> M3["ClasificadorDeObras<br/>ResNet18"] --> M4["Grad-CAM"]
        M1 --> M5["ExtractorDePaleta<br/>K-Means CIE Lab"]
    end
    subgraph ROSTRO["Rama del visitante"]
        F1["CapturaDeFrame<br/>cada 400 ms"] --> F2["AnalizarGestosFaciales<br/>YuNet + FER+"]
    end
    subgraph REGLAS["Reglas e integración"]
        R1["ResultadoProceso"] --> R2["Coincidencia<br/>1 − abs(v_obra − v_visitante) / 2"] --> R3["Integración (JSON)"]
    end
    subgraph DATOS["Google Drive"]
        D1[("dataset usado/<br/>positivo · negativo")]
        D2[("precarga-sin-analizar")]
        D3[("resultados_cnn/")]
    end
    U1 --> G1
    U2 --> G1
    G2 --> A1
    G1 -. en vivo .-> A2
    G6 --> A4 --> D2
    A1 --> M1
    A1 --> F1
    A2 --> F2
    M3 --> R1
    M5 --> R1
    F2 --> R1
    R3 --> G3
    D1 -. entrenamiento .-> M3
    D3 -. pesos .-> M3
```

## Módulos

| Módulo | Entrada | Salida | Técnica |
|---|---|---|---|
| ImageLoader | bytes JPG/PNG (≤ 10 MB, ≥ 32 px) | matriz RGB | Pillow + corrección EXIF |
| PreprocesamientoImagen | RGB | tensor `[1, 3, 224, 224]` | redimensionado + normalización de ImageNet |
| ClasificadorDeObras | tensor | 2 logits (negativo, positivo) | ResNet18 preentrenada; se ajustan `layer4` y la capa final |
| Grad-CAM | tensor + clase | mapa de calor y regiones | gradientes de `layer4` |
| ExtractorDePaleta | RGB | 6 colores HEX con su proporción | K-Means en CIE Lab |
| CapturaDeFrame | cámara del navegador | fotogramas con marca de tiempo | getUserMedia, 1 cada 400 ms |
| AnalizarGestosFaciales | fotograma | cajas, 5 puntos, 8 emociones, gestos, aptitud | YuNet + FER+ (ONNX) |
| ResultadoProceso | salidas crudas | resultados validados + excepciones | softmax, umbrales, calidad, valencia |
| Integración | todo lo anterior | JSON para la GUI | coincidencia e interpretación en texto |
