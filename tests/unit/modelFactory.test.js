/* @vitest-environment jsdom */
import { beforeEach, describe, expect, it } from 'vitest';
import { LMLModelFactory } from '../../src/algorithms/modelFactory';
import { ensureLocalStorage } from '../setup/browserPolyfills';

describe('LMLModelFactory', () => {
  beforeEach(() => {
    ensureLocalStorage();
    localStorage.clear();
  });

  it('creates models by type', () => {
    expect(LMLModelFactory.createModel('sequential').getAlgorithmName()).toBe('LMLSequential');
    expect(LMLModelFactory.createModel('knn').getAlgorithmName()).toBe('KNN');
    expect(LMLModelFactory.createModel('naive-bayes').getAlgorithmName()).toBe('NaiveBayes');
  });

  it('loads blank model when no persisted model exists', async () => {
    localStorage.removeItem('lmlModel');

    const loaded = await LMLModelFactory.load();
    expect(loaded.loadedFrom).toBe('blank');
    expect(loaded.lmlModel.getAlgorithmName()).toBe('LMLSequential');
  });

  it('loads model from localStorage metadata', async () => {
    localStorage.setItem('lmlModel', JSON.stringify({
      model: {
        metadata: {
          modelAlgorithm: 'naive-bayes',
          labels: ['A'],
          hyperparameters: { varianceSmoothing: 1e-9, classPriorSmoothing: 1 }
        },
        parameters: {
          labels: ['A'],
          counts: [1],
          means: [[0]],
          variances: [[1]],
          featureDim: 1,
          totalExamples: 1
        }
      },
      data: { name: 'dummy' },
      encoder: 'numerical'
    }));

    const loaded = await LMLModelFactory.load();
    expect(loaded.loadedFrom).toBe('localstorage');
    expect(loaded.lmlModel.getAlgorithmName()).toBe('NaiveBayes');
  });
});
