import * as tf from '@tensorflow/tfjs';
import { beforeAll } from 'vitest';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});
