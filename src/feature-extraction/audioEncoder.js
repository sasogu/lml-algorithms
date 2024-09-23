import * as tf from '@tensorflow/tfjs';

function base64ToArrayBuffer(base64) {
    let binaryString = atob(base64);
    let len = binaryString.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

async function getAudioData(base64Audio) {
    let arrayBuffer = base64ToArrayBuffer(base64Audio);
    let audioContext = new AudioContext();

    // Decodifica los datos de audio
    let audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Obtén el canal 0 (por ejemplo, el canal izquierdo en caso de ser estéreo)
    let channelData = audioBuffer.getChannelData(0);  // Datos de audio en formato Float32Array

    return channelData;
}

function hammingWindow(size) {
    let window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (size - 1));
    }
    return window;
}

async function calculateFFT(base64Audio) {    
   // Paso 1 y 2: Obtén los datos de audio
   let audioData = await getAudioData(base64Audio);
    
   // Asegúrate de que haya suficientes muestras (al menos 512)
   if (audioData.length < 512) {
       throw new Error("El audio no tiene suficientes muestras para aplicar una ventana de Hamming de 512.");
   }

   // Paso 3: Aplica la ventana de Hamming
   let hamming = hammingWindow(512);

   // Tomamos las primeras 512 muestras del audio
   let windowedAudio = audioData.slice(0, 512).map((sample, index) => sample * hamming[index]);

   // Convierte los datos de audio con ventana a un tensor de TensorFlow
   let audioTensor = tf.tensor1d(windowedAudio);
   
   // Calcula la FFT
   let fftResult = tf.spectral.rfft(audioTensor);

   // Opcional: Convierte el resultado complejo en magnitudes
   let magnitude = tf.abs(fftResult);

   // Devuelve el resultado como un array
   return magnitude;
}

export function audioEncoder(samples) {

    let promises = [];

    for (let sample of samples) {
        promises.push(calculateFFT(sample));
    }

    return Promise.all(promises).then(features =>{
        return tf.stack(features);
    });

       
}