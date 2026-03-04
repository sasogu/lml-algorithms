import { describe, expect, it } from 'vitest';
import { bowEncoder, buildVocabulary, setVocabulary } from '../../src/feature-extraction/bowEncoder';
import { numericalEncoder } from '../../src/feature-extraction/numerical';

describe('feature extraction encoders', () => {
  it('builds vocabulary and encodes BoW vectors', async () => {
    const vocabulary = buildVocabulary(['Hello world!', 'world test']);
    expect(vocabulary).toEqual(['hello', 'world', 'test']);

    const encoded = await bowEncoder(['hello test', 'world world']);
    expect(await encoded.array()).toEqual([
      [1, 0, 1],
      [0, 2, 0]
    ]);

    encoded.dispose();
  });

  it('uses externally provided vocabulary', async () => {
    setVocabulary(['red', 'blue']);

    const encoded = await bowEncoder(['red red blue']);
    expect(await encoded.array()).toEqual([[2, 1]]);

    encoded.dispose();
  });

  it('encodes CSV numeric rows to tensor', async () => {
    const tensor = await numericalEncoder(['1,2.5,3', '4,5,6']);
    expect(tensor.shape).toEqual([2, 3]);
    expect(await tensor.array()).toEqual([
      [1, 2.5, 3],
      [4, 5, 6]
    ]);

    tensor.dispose();
  });
});
