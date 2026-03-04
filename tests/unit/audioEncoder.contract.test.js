import { beforeAll, describe, expect, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';

const transferRecognizer = {
  dataset: { examples: {} },
  collectExample: vi.fn(async (label) => {
    transferRecognizer.dataset.examples = {
      sample1: {
        label,
        spectrogram: { data: [0, 0, 0, 0] }
      }
    };
    return { ok: true };
  })
};

const baseRecognizer = {
  ensureModelLoaded: vi.fn(async () => {}),
  createTransfer: vi.fn(() => transferRecognizer),
  modelInputShape: vi.fn(() => [null, 2, 2, 1]),
  recognize: vi.fn(async () => ({ embedding: tf.tensor2d([[0.1, 0.2, 0.3, 0.4]]) }))
};

const playRawAudioMock = vi.fn(() => 'played');
const createMock = vi.fn(() => baseRecognizer);

vi.mock('lml-speech-commands', () => ({
  create: createMock,
  utils: {
    playRawAudio: playRawAudioMock
  }
}));

describe('audioEncoder contract', () => {
  let collectExample;
  let audioEncoder;
  let playRawAudio;

  beforeAll(async () => {
    const mod = await import('../../src/feature-extraction/audioEncoder');
    collectExample = mod.collectExample;
    audioEncoder = mod.audioEncoder;
    playRawAudio = mod.playRawAudio;

    await new Promise(resolve => setTimeout(resolve, 0));
  });

  it('delegates playRawAudio to speechCommands utils', () => {
    expect(playRawAudio({ raw: true })).toBe('played');
    expect(playRawAudioMock).toHaveBeenCalled();
  });

  it('collects one example and removes temporary item from transfer dataset', async () => {
    const example = await collectExample('yes');

    expect(example.label).toBe('yes');
    expect(Object.keys(transferRecognizer.dataset.examples)).toHaveLength(0);
    expect(transferRecognizer.collectExample).toHaveBeenCalledWith('yes', { includeRawAudio: true });
  });

  it('encodes audio spectrogram examples into stacked embeddings', async () => {
    const output = await audioEncoder([
      { spectrogram: { data: [0, 0, 0, 0] } },
      { spectrogram: { data: [1, 1, 1, 1] } }
    ]);

    expect(output.shape).toEqual([2, 4]);
    output.dispose();
  });
});
