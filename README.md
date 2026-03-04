# lml-algorithms

Librería de algoritmos de Machine Learning y extractores de características usada por [LearningML](https://learningml.org).

## Qué incluye

- Algoritmos de clasificación:
  - `LMLSequential` (red neuronal feed-forward con TensorFlow.js)
  - `KNN`
  - `NaiveBayes` gaussiano
- Extracción de características:
  - Numérica (`numericalEncoder`)
  - Texto tipo Bag of Words (`bowEncoder`)
  - Audio (`audioEncoder`)
  - Imagen con MobileNet (`getMobilenetEncoder`)
- Utilidades para preparación de datos, validación y matriz de confusión.

## Instalación

Este repositorio usa JavaScript + TensorFlow.js y está preparado para ejecutarse con `bun` o `npm`.

```bash
bun install
```

Alternativa:

```bash
npm install
```

## API principal

Exportada desde `src/index.js`:

- Encoders: `getMobilenetEncoder`, `numericalEncoder`, `bowEncoder`, `audioEncoder`
- Audio helpers: `collectExample`, `playRawAudio`
- Algoritmos: `LMLSequential`, `KNN`, `NaiveBayes`, `LMLModelFactory`
- Utilidades: `confusionMatrix`, `transformObjectToMapWithTensors`, `combineMapsOfTensors`, `extendArraysInObject`

## Tests

La estrategia de tests está definida en [`AGENTS.md`](./AGENTS.md) y se implementa con **Vitest**.

### Política de ejecución recomendada

- `npm test` (`vitest run`) es la ejecución de referencia en CI.
- `bun test` está soportado para desarrollo local rápido.
- El test de contrato de `mobilenet` se marca como `skip` en el runner nativo de Bun porque su sistema de mocks no aísla módulos de `tfjs-core` igual que Vitest; ese contrato sí se valida en `npm test`.

### Ejecutar todos los tests

```bash
bun test
```

### Ejecutar solo unitarios

```bash
bun run test:unit
```

### Ejecutar solo integración

```bash
bun run test:integration
```

### Modo watch

```bash
bun run test:watch
```

### Con npm

```bash
npm test
npm run test:unit
npm run test:integration
```

## Estructura de tests

- `tests/unit`: utilidades, encoders, factoría de modelos y tests de contrato con mocks para dependencias de navegador/modelos externos.
- `tests/integration`: flujos completos de entrenamiento, clasificación y persistencia (`localStorage`) para los algoritmos.
- `tests/setup/vitest.setup.js`: inicialización de backend de TensorFlow para tests.

## Licencia

GPL-3.0-or-later.
