# Protocolo de evaluación con visitantes reales

Objetivo: medir el desempeño del **sistema completo** (no solo de la CNN) con personas reales, como pide el objetivo 3 de la rúbrica.

## 1. Preguntas de evaluación
| # | Pregunta | Métrica |
|---|---|---|
| E1 | ¿El sistema detecta el rostro del visitante en condiciones reales? | % de pruebas con rostro válido · % de fotogramas con rostro |
| E2 | ¿La expresión detectada (FER+) corresponde a lo que la persona dice sentir? | accuracy en 8 emociones y en valencia (neg/neu/pos) · κ de Cohen · IC 95 % (Wilson) |
| E3 | ¿La CNN clasifica las obras como las perciben las personas? (obras no vistas) | accuracy · F1 · κ · Spearman entre la valencia de la CNN y la media humana por obra |
| E4 | ¿La coincidencia calculada refleja la vivida? ¿Qué fórmula funciona mejor, v2 o v3? | MAE frente a la coincidencia reportada · Spearman con el acuerdo del visitante (1-5) |
| E5 | ¿El sistema responde a tiempo y es usable? | latencia (media, p95) · fps · puntaje SUS (0-100) |

## 2. Participantes y materiales
- **Participantes:** ≥ 10 (ideal 15) personas mayores de edad. Se identifican con un código anónimo `P01`, `P02`, … No se registran nombres en los datos.
- **Consentimiento:** cada persona firma el formato de la sección 6 antes de empezar.
- **Obras:** 6 por participante, tomadas de `precarga-sin-analizar` (la CNN no las vio). Hay que elegir antes 3 que la CNN clasifica como positivas y 3 como negativas (pestaña *Imágenes precargadas*). Se presentan en orden aleatorio.
- **Equipo:** portátil con cámara, buena iluminación frontal, fondo neutro, a 50-80 cm de la pantalla.

## 3. Procedimiento por participante (≈ 15 min)
1. Explicar el objetivo y firmar el consentimiento.
2. Para cada obra:
   1. **Opción A (en vivo):** pestaña *Obra y visitante* → cargar la obra → activar la cámara → *Iniciar observación* (6 s) mientras la persona mira la obra en grande → *Confirmar y analizar*.
   2. Mostrar el resultado y pedir que llene el formulario **Evaluación del sistema**: emoción que sintió, qué le transmitió la obra (−2 a +2) y acuerdo con la coincidencia (1-5) → *Guardar respuesta*.
   3. **Opción B (video):** grabar 6-10 s del rostro de frente mientras mira la obra y anotar sus respuestas en `protocolo_offline.csv` (sin acuerdo, porque no vio el resultado).
3. Al terminar, el cuestionario **SUS** (sección 5) en `sus.csv`.

> Las respuestas sobre la obra y la emoción se piden **antes** de comentar el resultado, para no influirlas.

## 4. Archivos (en `Drive/Proyecto-final-de-PDI/evaluacion/`)
| Archivo | Contenido |
|---|---|
| `registros_visitantes.jsonl` | una línea por prueba (la escriben la GUI y el notebook; no editar) |
| `protocolo_offline.csv` | `participante,obra,video,emocion_reportada,valencia_obra_reportada,acuerdo_coincidencia,comentario` |
| `videos/` | videos de la opción B (o una carpeta de fotos por prueba) |
| `obras/` | obras propias que no estén en `precarga-sin-analizar` (opcional) |
| `sus.csv` | `participante,q1,…,q10` (1 = muy en desacuerdo … 5 = muy de acuerdo) |

Plantillas vacías en la carpeta `evaluacion/` de este repositorio. Después de las pruebas, ejecuta la **sección 6** y luego la **sección 7** del notebook.

## 5. Cuestionario SUS (System Usability Scale)
1. Creo que me gustaría usar este sistema con frecuencia.
2. Encontré el sistema innecesariamente complejo.
3. Pensé que el sistema era fácil de usar.
4. Creo que necesitaría ayuda de una persona técnica para poder usarlo.
5. Encontré que las funciones del sistema estaban bien integradas.
6. Pensé que había demasiada inconsistencia en el sistema.
7. Imagino que la mayoría de las personas aprendería a usarlo muy rápido.
8. Encontré el sistema muy engorroso de usar.
9. Me sentí muy seguro(a) usando el sistema.
10. Necesité aprender muchas cosas antes de poder usarlo.

Puntaje = 2.5 × [Σ(ítems impares − 1) + Σ(5 − ítems pares)]. Un valor ≥ 68 se considera por encima del promedio.

## 6. Formato de consentimiento informado
> Yo, ____________________, acepto participar voluntariamente en la prueba del proyecto *Coincidencia emocional entre obra y visitante* (Procesamiento Digital de Imágenes, UAO). Entiendo que:
> - La cámara se usará para analizar mi expresión facial mientras observo obras de arte.
> - Con la opción en vivo **no se guardan imágenes de mi rostro**, solo valores numéricos y mis respuestas.
> - Con la opción de video, la grabación se usa solo para esta evaluación y se borra al terminar el semestre.
> - Mis datos se identifican con un código anónimo y se usan solo con fines académicos.
> - Puedo retirarme en cualquier momento sin dar explicaciones.
>
> Código asignado: ______ Firma: ____________________ Fecha: ____________
