import * as tf from '@tensorflow/tfjs-core';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import {
    getFormattedModelResult,
    getSuffledAndSplittedTensorsAndValidationData
} from './util';

export class NaiveBayes {

    constructor() {
        this.name = "model";
        this.model = null;
        this.labels = [];
        this.hyperparams = {
            varianceSmoothing: 1e-9,
            classPriorSmoothing: 1
        };
        this.features = new Map();
    }

    getAlgorithmName() {
        return "NaiveBayes";
    }

    setHyperParameters(params) {
        this.hyperparams = {
            ...this.hyperparams,
            ...params
        };
    }

    serialize() {
        let metadata = {
            modelAlgorithm: 'naive-bayes',
            labels: this.labels,
            hyperparameters: this.hyperparams
        };

        return Promise.resolve({
            metadata,
            parameters: this.model
        });
    }

    saveToMemory() {
        return;
    }

    saveToDisk(datatype) {
        return this.serialize().then(artifacts => {
            const zip = new JSZip();

            zip.file("model.parameters", JSON.stringify(artifacts.parameters));
            zip.file("model.metadata", JSON.stringify(artifacts.metadata));
            zip.file("model.datatype", JSON.stringify(datatype));

            return zip.generateAsync({ type: "blob" });
        }).then(content => {
            saveAs(content, "lml-naive-bayes_" + datatype.name + ".mdl");
            return 'model saved to disk';
        });
    }

    saveToLocalstorage(datatype, encoder) {
        return this.serialize().then(serializedModel => {
            let lmlModel = {
                model: serializedModel,
                data: datatype,
                encoder: encoder
            };
            localStorage.setItem('lmlModel', JSON.stringify(lmlModel));

            return lmlModel;
        });
    }

    loadFromLocalstorage() {

        let lmlModel = JSON.parse(localStorage.getItem('lmlModel'));

        if ('parameters' in lmlModel.model && lmlModel.model.parameters) {
            this.model = lmlModel.model.parameters;
            this.labels = lmlModel.model.metadata.labels || [];
            this.hyperparams = {
                ...this.hyperparams,
                ...(lmlModel.model.metadata.hyperparameters || {})
            };
        }

        return Promise.resolve({
            lmlModel: this,
            lmlModelMetadata: lmlModel,
            loadedFrom: 'localstorage'
        });
    }

    loadFromZipContent(modelZipContent) {
        let promises = [];

        let name = Object.keys(modelZipContent.files)[0].split(".")[0];

        let parameters = modelZipContent.files[name + '.parameters'];
        let metadata = modelZipContent.files[name + '.metadata'];
        let datatype = modelZipContent.files[name + '.datatype'];

        promises.push(parameters.async("string").then(content => JSON.parse(content)));
        promises.push(metadata.async("string").then(content => JSON.parse(content)));
        promises.push(datatype.async("string").then(content => JSON.parse(content)));

        return Promise.all(promises).then(r => {
            let parametersData = r[0];
            let metadataData = r[1];
            let data = r[2];
            this.model = parametersData;
            this.labels = metadataData.labels || [];
            this.hyperparams = {
                ...this.hyperparams,
                ...(metadataData.hyperparameters || {})
            };

            return {
                lmlModel: this,
                lmlModelMetadata: {
                    model: {
                        parameters: parametersData,
                        metadata: metadataData
                    },
                    data: data
                },
                loadedFrom: 'disk'
            };
        });
    }

    _buildGaussianModel(inputRows, outputLabels, numOfClasses) {
        if (!inputRows.length) {
            return null;
        }

        const featureDim = inputRows[0].length;
        const counts = new Array(numOfClasses).fill(0);
        const sums = Array.from({ length: numOfClasses }, () => new Array(featureDim).fill(0));
        const squaredSums = Array.from({ length: numOfClasses }, () => new Array(featureDim).fill(0));

        for (let i = 0; i < inputRows.length; i++) {
            const classIdx = outputLabels[i];
            const row = inputRows[i];
            counts[classIdx] += 1;
            for (let d = 0; d < featureDim; d++) {
                const value = row[d];
                sums[classIdx][d] += value;
                squaredSums[classIdx][d] += value * value;
            }
        }

        const means = Array.from({ length: numOfClasses }, () => new Array(featureDim).fill(0));
        const variances = Array.from({ length: numOfClasses }, () => new Array(featureDim).fill(0));

        for (let c = 0; c < numOfClasses; c++) {
            const classCount = counts[c];
            if (classCount === 0) {
                continue;
            }
            for (let d = 0; d < featureDim; d++) {
                const mean = sums[c][d] / classCount;
                const variance = Math.max((squaredSums[c][d] / classCount) - (mean * mean), 0);
                means[c][d] = mean;
                variances[c][d] = variance;
            }
        }

        return {
            labels: this.labels,
            counts,
            means,
            variances,
            featureDim,
            totalExamples: inputRows.length
        };
    }

    train(features, percentageForValidation = 0) {
        console.log("Naive Bayes train ...");
        this.features = features;
        this.labels = Array.from(features.keys());
        let tensors = getSuffledAndSplittedTensorsAndValidationData(features, percentageForValidation);

        const inputRows = tensors.inputTensor.arraySync();
        const outputRows = tensors.outputTensor.arraySync();
        const outputLabels = outputRows.map(row => row.indexOf(1));

        this.model = this._buildGaussianModel(inputRows, outputLabels, this.labels.length);

        let info = "NO INFO";
        let result = getFormattedModelResult(this.model, info, tensors.validationData);
        return Promise.resolve(result);
    }

    classify(inputTensor) {
        if (!this.model) {
            inputTensor.dispose();
            return Promise.reject(new Error('Naive Bayes model is not trained yet.'));
        }

        const varianceSmoothing = Number(this.hyperparams.varianceSmoothing) || 0;
        const classPriorSmoothing = Number(this.hyperparams.classPriorSmoothing) || 0;

        return inputTensor.array().then(array => {
            const input = Array.isArray(array[0]) ? array[0] : array;
            const classCount = this.labels.length;
            const denominatorPrior = this.model.totalExamples + (classPriorSmoothing * classCount);
            const logScores = new Array(classCount).fill(0);

            for (let c = 0; c < classCount; c++) {
                const priorNumerator = this.model.counts[c] + classPriorSmoothing;
                const prior = denominatorPrior > 0 ? priorNumerator / denominatorPrior : 0;
                let logScore = Math.log(prior || Number.EPSILON);

                for (let d = 0; d < this.model.featureDim; d++) {
                    const mean = this.model.means[c][d];
                    const variance = this.model.variances[c][d] + varianceSmoothing + Number.EPSILON;
                    const diff = input[d] - mean;
                    logScore += -0.5 * Math.log(2 * Math.PI * variance) - ((diff * diff) / (2 * variance));
                }

                logScores[c] = logScore;
            }

            const maxLog = Math.max(...logScores);
            const expScores = logScores.map(score => Math.exp(score - maxLog));
            const sumExpScores = expScores.reduce((acc, value) => acc + value, 0) || 1;

            const results = this.labels.map((label, index) => [label, expScores[index] / sumExpScores]);
            results.sort((a, b) => b[1] - a[1]);
            inputTensor.dispose();
            return results;
        });
    }
}
