# AGENTS.md

## Contexto del repositorio
- Librería JS de algoritmos y extracción de características para `learningml.org`.
- Stack principal: TensorFlow.js (`@tensorflow/tfjs`, `@tensorflow/tfjs-core`, `@tensorflow/tfjs-converter`) y utilidades de navegador (`localStorage`, `File`, `Blob`, `document`, `file-saver`).
- Estado actual: no hay tests automatizados (`npm test` falla por placeholder).

## Objetivo de testing
- Detectar regresiones en entrenamiento, clasificación, serialización y utilidades de tensores.
- Mantener feedback rápido en PRs con tests unitarios estables y tests de integración deterministas.
- Cubrir explícitamente APIs propensas a error: conversión de datos, split validación, persistencia de modelos y clasificación.

## Estrategia eficaz (pirámide)
1. Unitarios (70-80%): funciones puras o casi puras de `src/algorithms/util.js`, `bowEncoder.js`, `numerical.js`, `modelFactory.js`.
2. Integración (20-30%): flujo completo `encoder -> train -> classify -> serialize/load`.
3. Smoke E2E técnico (mínimo): validar que cada algoritmo principal arranca, entrena y clasifica con dataset pequeño.

## Recomendación de framework y ejecución
- Runner: `vitest` (rápido, buen soporte ESM y mocks).
- Entorno:
  - `node` para lógica algorítmica y tensores.
  - `jsdom` para módulos que usan `document`, `File`, `Blob`, `localStorage`.
- Comandos objetivo:
  - `test:unit`: unitarios puros.
  - `test:integration`: flujos completos con datasets pequeños.
  - `test`: ejecución conjunta para CI.

## Matriz de pruebas priorizada

### 1) Unit tests críticos
- `src/algorithms/util.js`
  - `getFeatureAndNumOfClasses`: dimensiones correctas.
  - `getInputAndOutputTensors`: one-hot correcto y orden de etiquetas estable.
  - `splitData`: tamaños esperados, casos borde (`splitFraction` <=0, >=1).
  - `getSuffledAndSplittedTensorsAndValidationData`: retorno consistente con/sin validación.
  - `transformObjectToMapWithTensors`, `combineMapsOfTensors`, `extendArraysInObject`.
  - `confusionMatrix`: estructura Plotly correcta y liberación de tensores.
- `src/feature-extraction/bowEncoder.js`
  - tokenización estable, vocabulario y codificación BoW esperada.
- `src/feature-extraction/numerical.js`
  - parseo CSV numérico y shape final.
- `src/algorithms/modelFactory.js`
  - creación por tipo (`sequential`, `knn`, `naive-bayes`) y fallback de `load`.

### 2) Integración de algoritmos
- `KNN`
  - `train` con dataset mínimo separable, `classify` devuelve etiqueta dominante esperada.
  - `setHyperParameters({ K })` afecta resultado.
  - serialización/carga por `localStorage` mantiene predicciones.
- `NaiveBayes`
  - `train` + `classify` sobre dataset gaussiano simple.
  - hiperparámetros (`varianceSmoothing`, `classPriorSmoothing`) no rompen estabilidad numérica.
  - `serialize/loadFromLocalstorage/loadFromZipContent` conserva `labels` y modelo.
- `LMLSequential`
  - entrenamiento corto (pocas épocas, batch pequeño) y `classify` funcional.
  - `saveToMemory` y `load` conservan top-1 en muestra de control.

### 3) Encoders dependientes de navegador/modelos externos
- `audioEncoder.js`
  - tests de contrato con mock de `lml-speech-commands` (sin descargar modelos reales).
  - validar forma de embeddings y eliminación de muestras temporales.
- `mobilenet.js`
  - tests de contrato con mock de `tfconv.loadGraphModel`.
  - validar path de inicialización, shape final `[n, 1024]` y comportamiento de `getMobilenetEncoder(url)`.

## Datos de prueba y determinismo
- Usar fixtures pequeños y linealmente separables (2-3 clases, 2-6 features).
- Fijar seeds cuando sea posible (`tf.util.shuffle` puede introducir ruido).
- Evitar asserts sobre precisión exacta de entrenamiento largo; preferir:
  - dimensiones y formato de salida,
  - etiqueta top-1 esperada en ejemplos simples,
  - monotonicidad básica (score clase correcta > clase incorrecta en casos triviales).

## Criterios de calidad
- Cobertura mínima sugerida:
  - líneas: >= 80% en `src/algorithms/util.js` y >= 70% global.
  - ramas: >= 65% global.
- Cero tests flaky permitidos en CI.
- Cada bug corregido debe venir con test de regresión.

## Mocks y utilidades recomendadas
- Mockear `file-saver` (`saveAs`) y `JSZip` cuando se pruebe persistencia.
- Mockear `localStorage` (o usar implementación de `jsdom`).
- Mockear `tf.loadLayersModel` y `tf.io.browserFiles` en tests de carga.
- Añadir helper para comprobar fugas:
  - capturar `tf.memory().numTensors` antes/después y permitir delta controlado.

## Riesgos actuales a blindar con tests
- Manejo de bordes en split/validación y datasets vacíos.
- Dependencia de APIs browser en entorno Node.
- Posibles fugas de memoria por tensores no liberados.
- Persistencia (JSON/weights/features) inconsistente entre algoritmos.

## Política de PR para tests
1. Si se toca `src/algorithms/*`, añadir o ajustar unit tests del módulo afectado.
2. Si se toca serialización/carga, añadir integración de round-trip.
3. Si se toca encoder browser (`audio`, `mobilenet`), incluir test de contrato con mocks.
4. No se aprueban cambios en lógica de entrenamiento sin al menos un caso de integración que entrene y clasifique.
