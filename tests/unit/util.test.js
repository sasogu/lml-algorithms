import { afterEach, describe, expect, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
  combineMapsOfTensors,
  confusionMatrix,
  extendArraysInObject,
  getFeatureAndNumOfClasses,
  getInputAndOutputTensors,
  getSuffledAndSplittedTensorsAndValidationData,
  splitData,
  transformObjectToMapWithTensors
} from '../../src/algorithms/util';

afterEach(() => {
  tf.disposeVariables();
});

describe('algorithms/util', () => {
  it('gets feature dimension and number of classes', () => {
    const features = new Map([
      ['cat', tf.tensor2d([[1, 2], [3, 4]])],
      ['dog', tf.tensor2d([[5, 6]])]
    ]);

    const result = getFeatureAndNumOfClasses(features);
    expect(result).toEqual({ featureDim: 2, numOfClasses: 2 });
  });

  it('builds input and one-hot output tensors from map', async () => {
    const features = new Map([
      ['cat', tf.tensor2d([[1, 0], [0, 1]])],
      ['dog', tf.tensor2d([[2, 2]])]
    ]);

    const { inputTensor, outputTensor } = getInputAndOutputTensors(features);
    expect(inputTensor.shape).toEqual([3, 2]);
    expect(outputTensor.shape).toEqual([3, 2]);

    expect(await outputTensor.array()).toEqual([
      [1, 0],
      [1, 0],
      [0, 1]
    ]);

    inputTensor.dispose();
    outputTensor.dispose();
  });

  it('splits data and returns validation tensors', () => {
    const input = tf.tensor2d([[1], [2], [3], [4]]);
    const output = tf.tensor2d([[1, 0], [1, 0], [0, 1], [0, 1]]);

    const [train, validation] = splitData(input, output, 0.5);

    expect(train.tensor_inputs_data.shape[0]).toBe(2);
    expect(train.tensor_outputs_data.shape[0]).toBe(2);
    expect(validation[0].shape[0]).toBe(2);
    expect(validation[1].shape[0]).toBe(2);

    train.tensor_inputs_data.dispose();
    train.tensor_outputs_data.dispose();
    validation[0].dispose();
    validation[1].dispose();
    input.dispose();
    output.dispose();
  });

  it('keeps all samples as training when split is invalid', () => {
    const input = tf.tensor2d([[1], [2], [3]]);
    const output = tf.tensor2d([[1, 0], [1, 0], [0, 1]]);

    const [train, validation] = splitData(input, output, 0);

    expect(validation).toBeNull();
    expect(train.tensor_inputs_data).toBe(input);
    expect(train.tensor_outputs_data).toBe(output);
  });

  it('returns tensors and validation data according to percentage', () => {
    const features = new Map([
      ['cat', tf.tensor2d([[1, 1], [1, 2]])],
      ['dog', tf.tensor2d([[9, 9], [9, 8]])]
    ]);

    const result = getSuffledAndSplittedTensorsAndValidationData(features, 50);

    expect(result.inputTensor.shape[0]).toBe(2);
    expect(result.outputTensor.shape[0]).toBe(2);
    expect(result.validationData).not.toBeNull();

    result.inputTensor.dispose();
    result.outputTensor.dispose();
    result.validationData[0].dispose();
    result.validationData[1].dispose();
  });

  it('transforms plain object into map of tensors', async () => {
    const map = transformObjectToMapWithTensors({
      a: [[1, 2]],
      b: [[3, 4]]
    });

    expect(map.size).toBe(2);
    expect(await map.get('a').array()).toEqual([[1, 2]]);
    expect(await map.get('b').array()).toEqual([[3, 4]]);
  });

  it('combines maps by concatenating matching keys', async () => {
    const map1 = new Map([
      ['a', tf.tensor2d([[1, 2]])],
      ['only1', tf.tensor2d([[7, 8]])]
    ]);
    const map2 = new Map([
      ['a', tf.tensor2d([[3, 4]])],
      ['only2', tf.tensor2d([[9, 9]])]
    ]);

    const combined = combineMapsOfTensors(map1, map2);
    expect(await combined.get('a').array()).toEqual([[1, 2], [3, 4]]);
    expect(await combined.get('only1').array()).toEqual([[7, 8]]);
    expect(await combined.get('only2').array()).toEqual([[9, 9]]);

    combined.get('a').dispose();
  });

  it('extends arrays in object with zeros', () => {
    const out = extendArraysInObject({ a: [[1, 2], [3, 4]] }, 3);
    expect(out).toEqual({
      a: [
        [1, 2, 0, 0, 0],
        [3, 4, 0, 0, 0]
      ]
    });
  });

  it('builds confusion matrix plotly data and disposes validation tensors', async () => {
    const validationDataset = {
      tensor_inputs: tf.tensor2d([[0], [1], [2], [3]]),
      tensor_outputs: tf.tensor2d([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
      ])
    };

    const model = {
      labels: ['cat', 'dog'],
      classify: vi.fn(async input => {
        const v = (await input.array())[0][0];
        input.dispose();
        return v < 2 ? [['cat', 1]] : [['dog', 1]];
      })
    };

    const result = await confusionMatrix(validationDataset, model);

    expect(result.layout.title).toBe('Confusion matrix');
    expect(result.data[0].type).toBe('heatmap');
    expect(result.data[0].z).toEqual([
      [2, 0],
      [0, 2]
    ]);
    expect(model.classify).toHaveBeenCalledTimes(4);
  });
});
