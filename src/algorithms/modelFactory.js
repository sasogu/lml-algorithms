import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import {
    getFormattedModelResult,
    getFeatureAndNumOfClasses,
    getSuffledAndSplittedTensorsAndValidationData
} from './util';

import { LMLSequential } from './sequential';
import { KNN } from './knn';

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

export class LMLModelFactory {

    static createModel(type) {
        this.labels = [];

        if (type == 'sequential') {
            return new LMLSequential;
        }

        if (type == 'knn') {
            return new KNN;
        }
    }

    /**
     * Este método crea un modelo mediante tres procedimientos distintos:
     * 
     * 1) si el modo es autonomous y se ha cargado un archivo de modelo desde 
     *    la GUI de Scratch, es decir, existe `modelZipContent`, entonces se
     *    crea un modelo con las característica del contenido de modelZipContent.
     * 2) si el modo no es autonomous  y en el localStorage hay un item lmlModelMetadata,
     *    se crea un modelo con las características del contenido de ese item,
     * 3) en otro caso, se crea un nuevo modelo por defecto (sequential)
     * 
     * @param {*} mode 
     * @param {*} modelZipContent 
     * @returns 
     */
    static load(mode, modelZipContent = null) {

        let metadata;

        if (mode == 'autonomous' && modelZipContent != null) {
            let name = Object.keys(modelZipContent.files)[0].split(".")[0];
            metadata = modelZipContent.files[name + '.metadata'];
            return metadata.async("string").then(content => {
                let lmlModelMetadata = JSON.parse(content);
                let model = LMLModelFactory.createModel(lmlModelMetadata.model.modelAlgorithm);
                return model.loadForAutomousMode(modelZipContent);
            })
        } else if (mode != 'autonomous' && localStorage.getItem('lmlModelMetadata') != null) {
            metadata = JSON.parse(localStorage.getItem('lmlModelMetadata'));
            let model = LMLModelFactory.createModel(metadata.model.modelAlgorithm);
            return model.loadForLMLMode();
        } else {
            metadata = {
                model: {
                    modelAlgorithm: 'sequential',
                    hyperparameters: {
                        learningRate: 0.001,
                        batchSize: 10,
                        epochs: 20
                    },
                    labels: []
                },
                status: {
                    modelEditor: 'text',
                    modelName: 'undefined'
                },
                loadedFrom: false
            }

            return Promise.resolve({
                model: LMLModelFactory.createModel(metadata.model.modelAlgorithm),
                lmlModelMetadata: metadata
            })
        }

    }
}