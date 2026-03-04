import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { LMLSequential } from '../../src/algorithms/sequential';

describe('integration: LMLSequential', () => {
  it('trains and classifies with a small dataset', async () => {
    const model = new LMLSequential();
    model.setHyperParameters({ learningRate: 0.01, batchSize: 2, epochs: 8 });

    const features = new Map([
      ['A', tf.tensor2d([[0, 0], [0.1, 0], [0.2, 0.1], [0, 0.2]])],
      ['B', tf.tensor2d([[1, 1], [0.9, 1], [1, 0.9], [0.8, 1]])]
    ]);

    const result = await model.train(features, 25);

    expect(result.model).toBeTruthy();
    expect(result.info.history.loss.length).toBeGreaterThan(0);

    const prediction = await model.classify(tf.tensor2d([[0.05, 0.1]]));
    expect(prediction.length).toBe(2);
    expect(prediction[0][1]).toBeGreaterThanOrEqual(prediction[1][1]);

    const memory = await model.saveToMemory();
    expect(memory.jsonFile).toBeTruthy();
    expect(memory.weights).toBeInstanceOf(ArrayBuffer);
  });
});
