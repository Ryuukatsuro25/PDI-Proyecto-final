# Resultados de la última ejecución

- Fuente de datos: Google Drive (/content/drive/MyDrive/Proyecto Final PDI VERSION 3/dataset usado)
- Imágenes válidas: positivo 400, negativo 400, precargadas 300 · descartadas 0
- Partición: entrenamiento 640 · validación 80 · prueba 80
- Preprocesamiento elegido por validación: sin filtros PDI

## CNN de obras (conjunto de prueba)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Modelo B (elegido) | 0.750 | 0.750 | 0.750 | 0.750 |

Matriz de confusión del modelo elegido [[neg→neg, neg→pos], [pos→neg, pos→pos]]: [[30, 10], [10, 30]]

## Agrupamiento no supervisado

- Obras etiquetadas: k = 2, silueta = 0.342, ARI con las etiquetas = 0.013 (k = 2: 0.013)
- Precargadas: k = 2, silueta = 0.342

## Neutralidad facial

- Variante «corregido» (AUC 0.82, 122 fotos del equipo). τ = 0.70 maximiza el índice de Youden: exactitud balanceada 77% con τ = 0.5 → 81% con τ calibrado.
- original: detección 100%, AUC 0.799, τ 0.73, exactitud balanceada 0.749 → 0.781
- corregido: detección 100%, AUC 0.818, τ 0.70, exactitud balanceada 0.774 → 0.814
- corregido_clahe: detección 100%, AUC 0.807, τ 0.49, exactitud balanceada 0.797 → 0.814

### Evaluación independiente (fotos de prueba no usadas para elegir τ)

- Fotos: calibración 85 · prueba 37 · variante corregido · τ = 0.70
- Detección de rostro en prueba: 100%
- AUC prueba: 0.801 · exactitud balanceada τ=0.5: 0.737 · con τ calibrado: 0.816
- Matriz [[neutral bien, neutral→no], [no→neutral, no neutral bien]]: [[18, 0], [7, 12]]

## Figuras

![01_Conjunto_de_datos_del_equipo](01_Conjunto_de_datos_del_equipo.png)
![02_Preprocesamiento_PDI_filtrado_espacial_y_en_frecue](02_Preprocesamiento_PDI_filtrado_espacial_y_en_frecue.png)
![03_Ejemplo_de_predicción](03_Ejemplo_de_predicción.png)
![04_Modelo_B_guardado_Pérdida](04_Modelo_B_guardado_Pérdida.png)
![05_Matriz_de_confusión_Modelo_B_elegido](05_Matriz_de_confusión_Modelo_B_elegido.png)
![06_real_negativo_pred_positivo](06_real_negativo_pred_positivo.png)
![07_Obras_etiquetadas_k_2_silueta_0_34](07_Obras_etiquetadas_k_2_silueta_0_34.png)
![08_Obras_precargadas_k_2_silueta_0_34](08_Obras_precargadas_k_2_silueta_0_34.png)
![09_Obras_más_representativas_de_cada_grupo_precargada](09_Obras_más_representativas_de_cada_grupo_precargada.png)
![10_ROC_de_la_decisión_neutral_no_neutral](10_ROC_de_la_decisión_neutral_no_neutral.png)
![11_ROC_neutralidad_fotos_de_prueba_independientes](11_ROC_neutralidad_fotos_de_prueba_independientes.png)
