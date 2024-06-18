import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
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

    serialize() {
        return {
            modelAlgorithm: 'sequential',
            labels: this.labels,
            hyperParameters: this.hyperparams
        }
    }

    getName() {
        return 'sequential';
    }

    setHyperParameters(params) {
        this.hyperparams = params;
    }

    saveToDisk(name = 'model', lmlModelMetadata) {
        console.log("SAVING MODEL ZIP");
        return this.model.save(tf.io.withSaveHandler(async (artifacts) => {
            const zip = new JSZip();
            let jsonFile = {
                modelTopology: artifacts.modelTopology,
                format: artifacts.format,
                generatedBy: artifacts.generatedBy,
                convertedBy: artifacts.convertedBy,
                weightsManifest: [{
                    paths: [
                        './' + name + '.weights.bin'
                    ],
                    weights: artifacts.weightSpecs
                }
                ]
            }
            zip.file(name + ".json", JSON.stringify(jsonFile));
            const weights = new Uint8Array(artifacts.weightData);
            zip.file(name + ".weights.bin", weights);
            zip.file(name + ".metadata", JSON.stringify(lmlModelMetadata));

            return zip.generateAsync({ type: "blob" });
        })).then(content => {
            saveAs(content, "lml_" + name + ".mdl");
            return 'model saved to disk';
        });
    }

    saveToLocalstorage(){
        this.model.model.save('localstorage://lml');
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
        this.model = this.buildModel(dims.featureDim, dims.numOfClasses);

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
            let result = getFormattedModelResult(this.model, info, tensors.validationData);

            return result;
        });
    }


    load(mode, modelZipContent = null) {
        if (mode == 'autonomous') {
            return this.loadForAutomousMode(modelZipContent);
        } else {
            return this.loadForLMLMode();
        }
    }

    loadForAutomousMode(modelZipContent) {

        let promises = [];

        let name = Object.keys(modelZipContent.files)[0].split(".")[0];

        let jsonFile = modelZipContent.files[name + '.json'];
        let weightsFile = modelZipContent.files[name + '.weights.bin'];
        let metadata = modelZipContent.files[name + '.metadata'];

        promises.push(jsonFile.async("blob").then(blob => {
            const file = new File([blob], jsonFile.name, {
                type: blob.type
            });
            return file;
        }));
        promises.push(weightsFile.async("blob").then(blob => {
            const file = new File([blob], weightsFile.name, {
                type: blob.type
            });
            return file;
        }));

        promises.push(metadata.async("string").then(content => {
            return JSON.parse(content);
        }));

        return Promise.all(promises).then(r => {
            console.log(r);
            return tf.loadLayersModel(tf.io.browserFiles([r[0], r[1]]))
                .then(model => {
                    this.model = model;
                    return {
                        model: this,
                        lmlModelMetadata: r[2],
                        loadedFrom: 'disk'
                    };
                });
        })

    }

    loadForLMLMode() {
        if (!localStorage.getItem('lmlModelMetadata')) {
            return;
        }

        return tf.loadLayersModel('localstorage://lml').then(model => {
            this.model = model;
            return {
                model: this,
                lmlModelMetadata: JSON.parse(localStorage.getItem('lmlModelMetadata')),
                loadedFrom: 'localstorage'
            };
        });
    }

    classify(inputTensor, labels = null) {
        const prediction = this.model.predict(inputTensor);
        inputTensor.dispose();
        const predictions = prediction.dataSync();
        console.log(predictions);
        console.log(this.labels);

        // Si se ejecutó train antes de classify, this.labels está rellena, pero 
        // si se ejecutó classify sin llamar antes a train (esto pasa cuando se carga 
        // un modelo desde el localstorage) hay que pasar la lista de labels        
        let _labels = (labels == null) ? this.labels : labels
        const arr_predictions = Array.from(predictions);
        let results = [];
        for (let i = 0; i < arr_predictions.length; i++) {
            results.push([_labels[i], arr_predictions[i]]);
        }
        results.sort((a, b) => b[1] - a[1]);
        return new Promise((resolve, reject) => { resolve(results) });
    }
}