/* @vitest-environment jsdom */
import { beforeEach, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { KNN } from '../../src/algorithms/knn';
import { NaiveBayes } from '../../src/algorithms/naive-bayes';
import { ensureLocalStorage } from '../setup/browserPolyfills';

describe('integration: KNN + NaiveBayes', () => {
  beforeEach(() => {
    ensureLocalStorage();
    localStorage.clear();
  });

  it('trains KNN, classifies and keeps behavior after localStorage round-trip', async () => {
    const knn = new KNN();
    knn.setHyperParameters({ K: 1 });

    const features = new Map([
      ['cold', tf.tensor2d([[0, 0], [0.1, 0.2], [0.2, 0.1]])],
      ['hot', tf.tensor2d([[10, 10], [9.8, 10.2], [10.2, 9.9]])]
    ]);

    await knn.train(features);

    const predictionTensor = tf.tensor2d([[0.05, 0.1]]);
    const resultBefore = await knn.classify(predictionTensor);
    expect(resultBefore[0][0]).toBe('cold');

    await knn.saveToLocalstorage({ name: 'integration-knn', type: 'numeric' }, 'numerical');

    const restored = new KNN();
    await restored.loadFromLocalstorage();
    restored.setHyperParameters({ K: 1 });

    const predictionTensorAfter = tf.tensor2d([[0.05, 0.1]]);
    const resultAfter = await restored.classify(predictionTensorAfter);

    expect(resultAfter[0][0]).toBe(resultBefore[0][0]);
  });

  it('trains Naive Bayes, classifies and restores from localStorage', async () => {
    const nb = new NaiveBayes();
    nb.setHyperParameters({ varianceSmoothing: 1e-8, classPriorSmoothing: 1 });

    const features = new Map([
      ['left', tf.tensor2d([[-2, -1], [-1.5, -1.2], [-1.8, -0.8]])],
      ['right', tf.tensor2d([[2, 1], [1.8, 1.2], [2.2, 0.9]])]
    ]);

    await nb.train(features);

    const input = tf.tensor2d([[-1.9, -1.0]]);
    const resultBefore = await nb.classify(input);
    expect(resultBefore[0][0]).toBe('left');

    await nb.saveToLocalstorage({ name: 'integration-nb', type: 'numeric' }, 'numerical');

    const restored = new NaiveBayes();
    await restored.loadFromLocalstorage();
    const inputAfter = tf.tensor2d([[-1.9, -1.0]]);
    const resultAfter = await restored.classify(inputAfter);

    expect(resultAfter[0][0]).toBe(resultBefore[0][0]);
  });
});
