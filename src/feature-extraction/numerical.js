import * as tf from '@tensorflow/tfjs';

export function numericalEncoder(items) {
    return new Promise((resolve, reject) => {
        let features = [];
        for (let csv of items) {
            features.push(csv.split(",").map(v => parseFloat(v)));
        }
        resolve(tf.stack(features));
    });    
} 