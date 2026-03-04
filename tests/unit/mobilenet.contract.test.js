/* @vitest-environment jsdom */
import { describe, expect, it, vi } from 'vitest';

const isBunNativeRunner = typeof Bun !== 'undefined';

if (isBunNativeRunner) {
  describe.skip('mobilenet encoder contract (skipped on bun native runner)', () => {
    it('is validated in vitest runner', () => {});
  });
} else {
  describe('mobilenet encoder contract', () => {
    it('loads graph model from configured URL and returns [n,1024] embeddings', async () => {
      vi.resetModules();

      class Tensor {
        constructor(shape = []) {
          this.shape = shape;
        }

        data() {
          return Promise.resolve(new Float32Array([0]));
        }

        dispose() {}
      }

      const loadGraphModelMock = vi.fn();
      const predictMock = vi.fn(() => new Tensor([1, 1001]));
      const executeMock = vi.fn(() => new Tensor([1, 1, 1, 1024]));

      loadGraphModelMock.mockResolvedValue({
        predict: predictMock,
        execute: executeMock
      });

      vi.doMock('@tensorflow/tfjs-converter', () => ({
        loadGraphModel: loadGraphModelMock
      }));

      vi.doMock('@tensorflow/tfjs-core', () => ({
        Tensor,
        tidy: (fn) => fn(),
        zeros: (shape) => new Tensor(shape),
        cast: (tensor) => tensor,
        mul: (tensor) => tensor,
        add: (tensor) => tensor,
        reshape: (_tensor, shape) => new Tensor(shape),
        squeeze: () => new Tensor([1, 1024]),
        stack: (tensors) => new Tensor([tensors.length, 1, 1024]),
        image: {
          resizeBilinear: () => new Tensor([224, 224, 3])
        },
        browser: {
          fromPixels: () => new Tensor([224, 224, 3])
        }
      }));

      const originalCreateElement = document.createElement.bind(document);
      vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
        if (tagName !== 'img') {
          return originalCreateElement(tagName);
        }

        const img = { width: 0, height: 0, onload: null };
        Object.defineProperty(img, 'src', {
          set() {
            queueMicrotask(() => {
              if (typeof img.onload === 'function') {
                img.onload();
              }
            });
          }
        });

        return img;
      });

      const { getMobilenetEncoder } = await import('../../src/feature-extraction/mobilenet');
      const encoder = getMobilenetEncoder('https://cdn.learningml.org');

      const output = await encoder(['data:image/png;base64,AAA=', 'data:image/png;base64,BBB=']);

      expect(output.shape).toEqual([2, 1024]);
      expect(loadGraphModelMock).toHaveBeenCalledWith(
        'https://cdn.learningml.org/models/imagenet_mobilenet_v1_100_224_classification_1',
        { fromTFHub: true }
      );
      expect(executeMock).toHaveBeenCalled();

      vi.doUnmock('@tensorflow/tfjs-converter');
      vi.doUnmock('@tensorflow/tfjs-core');
    });
  });
}
