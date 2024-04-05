import * as tf from '@tensorflow/tfjs';
import { 
    getFormattedModelResult,
    getFeatureAndNumOfClasses,
    getSuffledAndSplittedTensorsAndValidationData
 } from './util';

/**
 * @typedef {Object} Dimensiones
 * @property {number} featureDim - dimensión del vector de características 
 * @property {number} numOfClasses - nº de clases
 */

/**
 * @typedef {Object} InOutTensors
 * @property {number} inputTensor - Tensor de shape = [n, featureDim], hay n ejemplares
 *                                  codificados con un vector de dimensión featureDim
 * @property {number} outputTensor - Tensor de shape = [n, numOfClasses], hay n ejemplares
 *                                   de vectores one-hot-encoded, cada uno de ellos representa 
 *                                   la clase a la que corresponde el ejemplar asociado.
 */

export class LMLSequential {

    constructor() {
        this.model = null;
        this.labels = [];
        this.hyperparams = {
            learningRate: 0.001,
            batchSize: 10,
            epochs: 20
        };
    }

    setHyperParameters(params) {
        this.hyperparams = params;
    }


    /**
     * 
     * Creamos un modelo consistente en una red neuronal feedforward con una capa de entrada,
     * una capa oculta y una capa de salida. 
     * 
     * @param {Number} inputDim - Dimensión de la capa de entrada (dimensión de los vectores de características)
     * @param {Number} outputDim - Dimensión de la capa de salida (nº de clases)
     * 
     * @returns {tf.Sequential}
     */
    buildModel(inputDim, outputDim) {

        let model = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: 200,
                    inputDim: inputDim,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 100,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    useBias: true
                }),
                tf.layers.dense({
                    units: outputDim,
                    kernelInitializer: 'varianceScaling',
                    useBias: false,
                    activation: 'softmax'
                })
            ]
        });

        // 0.001 es el learning rate
        //const optimizer = tf.train.adamax();
        const optimizer = tf.train.adam(this.hyperparams.learningRate)

        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    /**
     * 
     * @param {Map(String, tf.Tensor)} features - Mapa de las `features` que han resultado 
     *                                            del proceso de extracción de características
     * @param {InOutTensors} percentageForValidation - porcentaje de datos que serán usados para validar
     *                                                 el modelo
     * @returns {Promise()}
     */
    train(features, percentageForValidation = 0) {

        this.labels = Array.from(features.keys());
        let dims = getFeatureAndNumOfClasses(features);
        let tensors = getSuffledAndSplittedTensorsAndValidationData(features, percentageForValidation);     

        ////////////// SEQUENTIAL /////////////
        if (!this.model || (this.hyperparams.learningRate != 0.001)) {
            this.model = this.buildModel(dims.featureDim, dims.numOfClasses);
        }

        function onBatchEnd(batch, logs) {
            console.log('Accuracy', logs.acc);
        }

        //Train for 5 epochs with batch size of 32.
        return this.model.fit(tensors.inputTensor, tensors.outputTensor, {
            epochs: this.hyperparams.epochs,
            batchSize: this.hyperparams.batchSize,
            callbacks: { onBatchEnd },
            shuffle: true,
            //validationSplit: this.params.neural_network.validationSplit/100,
            validationData: tensors.validationData
        }).then(info => {
            this.model.save('indexeddb://lml-sequential-model');
            
            ////////////////////////////////////////
            
            let result = getFormattedModelResult(this.model, info, tensors.validationData);         
            
            return result;
        });
    }

    classify(inputTensor) {
        const prediction = this.model.predict(inputTensor);
        inputTensor.dispose();
        const predictions = prediction.dataSync();
        console.log(predictions);
        console.log(this.labels);
        const arr_predictions = Array.from(predictions);
        let results = [];
        for (let i = 0; i < arr_predictions.length; i++) {
            results.push([this.labels[i], arr_predictions[i]]);
        }
        results.sort((a, b) => b[1] - a[1]);
        return new Promise((resolve, reject) => { resolve(results) });
    }
}