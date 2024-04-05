import * as use from '@tensorflow-models/universal-sentence-encoder';

let encoder;
export function useEncoder(items) {

    if (encoder == null) {
        encoder = use.load();
    }

    return encoder.then(model => {
        return model.embed(items);
    })
}