import * as tf from '@tensorflow/tfjs';

let vocabulary;
// Función para limpiar y separar el texto en palabras
function tokenize(text) {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/);
}

// Función para construir el vocabulario a partir de los textos
function buildVocabulary(texts) {
    const _vocabulary = new Set();
    texts.forEach(text => {
        const tokens = tokenize(text);
        tokens.forEach(token => {
            if (token) _vocabulary.add(token);
        });
    });
    vocabulary = Array.from(_vocabulary);
    return vocabulary;
}

function setVocabulary(_vocabulary){
    vocabulary = _vocabulary;
}

// Función para codificar los textos en vectores BoW
function bowEncoder(texts) {    
    const vectors = texts.map(text => {
        const tokens = tokenize(text);
        const vector = Array(vocabulary.length).fill(0);
        tokens.forEach(token => {
            if (token) {
                const index = vocabulary.indexOf(token);
                if (index !== -1) parseFloat(vector[index]++);
            }
        });
        return vector;
    });

    return Promise.resolve(tf.stack(vectors));
}

export {
    buildVocabulary,
    setVocabulary,
    bowEncoder
}
