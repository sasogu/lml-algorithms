import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import {
    getFormattedModelResult,
    getFeatureAndNumOfClasses,
    getSuffledAndSplittedTensorsAndValidationData
} from './util';

/**
 * @typedef {Object} InOutTensors
 * @property {number} inputTensor  - Tensor de shape = [n, featureDim], hay n ejemplares
 *                                   codificados con un vector de dimensión featureDim
 * @property {number} outputTensor - Tensor de shape = [n, numOfClasses], hay n ejemplares
 *                                   de vectores one-hot-encoded, cada uno de ellos representa 
 *                                   la clase a la que corresponde el ejemplar asociado.
 */

export class LMLSequential {

    constructor() {
        this.name = "model";
        this.model = null;
        this.labels = [];
        this.hyperparams = {
            learningRate: 0.001,
            batchSize: 10,
            epochs: 20
        };
    }

    serialize() {

        function arrayBufferToBase64(buffer) {
            // Crear una cadena binaria a partir del array de bytes
            let binary = '';
            const bytes = new Uint8Array(buffer);
            const len = bytes.byteLength;
            for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(bytes[i]);
            }

            // Convertir la cadena binaria a una cadena en formato Base64
            return btoa(binary);
        }

        let metadata = {
            modelAlgorithm: 'sequential',
            labels: this.labels,
            hyperparameters: this.hyperparams
        };

        if (this.model != null) {
            return this.saveToMemory().then(artifacts => {
                return {
                    metadata: metadata,
                    jsonFile: artifacts.jsonFile,
                    weights: arrayBufferToBase64(artifacts.weights)
                }
            })
        } else {
            return Promise.resolve({
                metadata: metadata,
                jsonFile: null,
                weights: null,
            })
        }
    }

    setHyperParameters(params) {
        this.hyperparams = params;
    }

    saveToMemory() {
        return this.model.save(tf.io.withSaveHandler(artifacts => {
            let jsonFile = {
                modelTopology: artifacts.modelTopology,
                format: artifacts.format,
                generatedBy: artifacts.generatedBy,
                convertedBy: artifacts.convertedBy,
                weightsManifest: [{
                    paths: [
                        './' + this.name + '.weights.bin'
                    ],
                    weights: artifacts.weightSpecs
                }
                ]
            }

            return {
                jsonFile: jsonFile,
                weights: artifacts.weightData,
            }
        }));
    }

    saveToDisk(datatype) {
        console.log("SAVING MODEL ZIP");

        let metadata = {
            modelAlgorithm: 'sequential',
            labels: this.labels,
            hyperparameters: this.hyperparams
        };

        return this.saveToMemory().then(artifacts => {
            const zip = new JSZip();

            zip.file("model.json", JSON.stringify(artifacts.jsonFile));
            const weights = new Uint8Array(artifacts.weights);
            zip.file("model.weights.bin", weights);
            zip.file("model.metadata", JSON.stringify(metadata));
            zip.file("model.datatype", JSON.stringify(datatype));

            return zip.generateAsync({ type: "blob" });
        }).then(content => {
            saveAs(content, "lml_" + datatype.name + ".mdl");
            return 'model saved to disk';
        });
    }

    saveToLocalstorage(datatype) {
        return this.serialize().then(serializedModel => {
            let lmlModel = {
                model: serializedModel,
                data: datatype,
            };
            localStorage.setItem('lmlModel', JSON.stringify(lmlModel));

            return lmlModel;
        });
    }

    loadFromLocalstorage() {

        function jsonFileToFile(jsonFile) {
            const jsonString = JSON.stringify(jsonFile);
            const blob = new Blob([jsonString], { type: "application/json" });
            const file = new File([blob], "model.json", { type: "application/json" });

            return file;
        }

        function weightsToFile(weightsB64) {
            const binaryString = atob(weightsB64);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);

            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            const blob = new Blob([bytes.buffer], { type: 'application/octet-stream' });

            const file = new File([blob], 'model.weights.bin', {
                type: 'application/octet-stream',
                lastModified: Date.now()
            });

            return file;
        }

        console.log("LoadFromLocalStorage");
        let lmlModel = JSON.parse(localStorage.getItem('lmlModel'));
        let jsonFile = lmlModel.model.jsonFile;
        let weights = lmlModel.model.weights;
        let metadata = lmlModel.model.metadata;
        let data = lmlModel.data;
        if ('jsonFile' in lmlModel.model && lmlModel.model.jsonFile) {
            jsonFile = jsonFileToFile(jsonFile);
            weights = weightsToFile(weights);
            return tf.loadLayersModel(tf.io.browserFiles([jsonFile, weights]))
                .then(model => {
                    this.model = model;
                    this.labels = metadata.labels;
                    return {
                        lmlModel: this,
                        lmlModelMetadata: {
                            model: {
                                jsonFile, weights, metadata
                            },
                            data,
                        },
                        loadedFrom: 'localstorage'
                    };
                });
        } else {
            let lmlModel = {
                lmlModel: this,
                lmlModelMetadata: {
                    model: {
                        jsonFile, weights, metadata
                    },
                    data,
                },
                loadedFrom: 'localstorage'
            };
            return Promise.resolve(lmlModel);
        }

    }

    loadFromZipContent(modelZipContent) {

        let promises = [];

        let name = Object.keys(modelZipContent.files)[0].split(".")[0];

        let jsonFile = modelZipContent.files[name + '.json'];
        let weightsFile = modelZipContent.files[name + '.weights.bin'];
        let metadata = modelZipContent.files[name + '.metadata'];
        let datatype = modelZipContent.files[name + '.datatype'];

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

        promises.push(datatype.async("string").then(content => {
            return JSON.parse(content);
        }));

        return Promise.all(promises).then(r => {
            console.log(r);
            let jsonFile = r[0];
            let weights = r[1];
            let metadata = r[2];
            let data = r[3];
            return tf.loadLayersModel(tf.io.browserFiles([jsonFile, weights]))
                .then(model => {
                    this.model = model;
                    this.labels = metadata.labels;
                    return {
                        lmlModel: this,
                        lmlModelMetadata: {
                            model: {
                                jsonFile, weights, metadata
                            },
                            data,
                        },
                        loadedFrom: 'disk'
                    };
                });
        })

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
     * @param {Map(String, tf.Tensor)} features      - Mapa de las `features` que han resultado 
     *                                                 del proceso de extracción de características
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

    classify(inputTensor, labels = null) {
        try {
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
            return Promise.resolve(results);
        } catch (error) {
            return Promise.reject(error);
        }
        
    }
}