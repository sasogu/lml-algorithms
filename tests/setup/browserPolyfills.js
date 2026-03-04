function createMemoryStorage() {
  const store = new Map();

  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
    removeItem(key) {
      store.delete(key);
    },
    clear() {
      store.clear();
    },
    key(index) {
      return Array.from(store.keys())[index] ?? null;
    },
    get length() {
      return store.size;
    }
  };
}

export function ensureLocalStorage() {
  if (typeof globalThis.localStorage === 'undefined') {
    globalThis.localStorage = createMemoryStorage();
  }
}

export function ensureDocumentForImages() {
  if (typeof globalThis.document === 'undefined') {
    globalThis.document = {
      createElement(tagName) {
        if (tagName !== 'img') {
          return { tagName };
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
      }
    };
  }
}
