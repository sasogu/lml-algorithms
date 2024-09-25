import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';

let transferRecognizer;

(async function initSpeechCommands() {

    const baseRecognizer = speechCommands.create('BROWSER_FFT');
    await baseRecognizer.ensureModelLoaded();

    transferRecognizer = baseRecognizer.createTransfer("audio-model");

    console.log("transferRecognizer");
    console.log(transferRecognizer);
})()

export function collectExample(label) {

    let options = {
        includeRawAudio: true
    };

    return transferRecognizer.collectExample(label, options).then(spectrogram => {
        console.log(spectrogram);
        console.log(transferRecognizer);

        const lastKey = Object.keys(transferRecognizer.dataset.examples).pop();

        // Obtener el valor correspondiente a esa clave
        const lastValue = transferRecognizer.dataset.examples[lastKey];

        return lastValue;
    })
}
export function audioEncoder(samples) {

    let promises = [];

    for (let sample of samples) {
        promises.push(calculateFFT(sample));
    }

    return Promise.all(promises).then(features => {
        return tf.stack(features);
    });


}