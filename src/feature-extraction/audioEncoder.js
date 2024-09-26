import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';

let baseRecognizer;
let transferRecognizer;

(async function initSpeechCommands() {

    baseRecognizer = speechCommands.create('BROWSER_FFT');
    await baseRecognizer.ensureModelLoaded();

    transferRecognizer = baseRecognizer.createTransfer("audio-model");

    console.log("transferRecognizer");
    console.log(transferRecognizer);
})()

export function collectExample(label) {

    let options = {
        includeRawAudio: false,
    };

    return transferRecognizer.collectExample(label, options);
}
export function audioEncoder(spectrograms) {

    let embeddingsPromise = [];
    for (let spec of spectrograms) {
        const x = tf.tensor4d(
            spec, [1].concat(baseRecognizer.modelInputShape().slice(1)));
        
        let embedding = baseRecognizer.recognize(x, { includeEmbedding: true })
            .then(result => {
                return result.embedding.reshape([result.embedding.shape[1]]);
            });

        embeddingsPromise.push(embedding);
    }

    return Promise.all(embeddingsPromise).then(features => {
        return tf.stack(features);
    })
}