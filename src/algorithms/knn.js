import * as tf from '@tensorflow/tfjs-core';
import { 
    getFormattedModelResult, 
    getSuffledAndSplittedTensorsAndValidationData } from './util';

function concatWithNulls(ndarray1, ndarray2) {
    if (ndarray1 == null && ndarray2 == null) {
        return null;
    }
    if (ndarray1 == null) {
        return ndarray2.clone();
    } else if (ndarray2 === null) {
        return ndarray1.clone();
    }
    return tf.concat([ndarray1, ndarray2], 0);
}

function topK(values, k) {
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort((a, b) => {
        return b.value - a.value;
    });
    const topkValues = new Float32Array(k);
    const topkIndices = new Int32Array(k);
    for (let i = 0; i < k; i++) {
        topkValues[i] = valuesAndIndices[i].value;
        topkIndices[i] = valuesAndIndices[i].index;
    }
    return { values: topkValues, indices: topkIndices };
}


class KNNClassifier {
    // The full concatenated dataset that is constructed lazily before making a
    // prediction.
    trainDatasetMatrix;

    // Individual class datasets used when adding examples. These get concatenated
    // into the full trainDatasetMatrix when a prediction is made.
    classDatasetMatrices = {};
    classExampleCount = {};

    exampleShape;
    labelToClassId = {};
    nextClassId = 0;

    /**
     * Adds the provided example to the specified class.
     */
    addExample(example, label) {
        if (this.exampleShape == null) {
            this.exampleShape = example.shape;
        }
        if (!tf.util.arraysEqual(this.exampleShape, example.shape)) {
            throw new Error(
                `Example shape provided, ${example.shape} does not match ` +
                `previously provided example shapes ${this.exampleShape}.`);
        }

        this.clearTrainDatasetMatrix();

        if (!(label in this.labelToClassId)) {
            this.labelToClassId[label] = this.nextClassId++;
        }

        tf.tidy(() => {
            const normalizedExample =
                this.normalizeVectorToUnitLength(tf.reshape(example, [example.size]));
            const exampleSize = normalizedExample.shape[0];

            if (this.classDatasetMatrices[label] == null) {
                this.classDatasetMatrices[label] =
                    tf.reshape(normalizedExample, [1, exampleSize]);
            } else {
                const newTrainLogitsMatrix =
                    tf.concat([
                        tf.reshape(this.classDatasetMatrices[label],
                            [this.classExampleCount[label], exampleSize]),
                        tf.reshape(normalizedExample, [1, exampleSize])
                    ], 0);

                this.classDatasetMatrices[label].dispose();
                this.classDatasetMatrices[label] = newTrainLogitsMatrix;
            }

            tf.keep(this.classDatasetMatrices[label]);

            if (this.classExampleCount[label] == null) {
                this.classExampleCount[label] = 0;
            }
            this.classExampleCount[label]++;
        });
    }

    /**
     * This method return distances between the input and all examples in the
     * dataset.
     *
     * @param input The input example.
     * @returns cosine similarities for each entry in the database.
     */
    similarities(input) {
        return tf.tidy(() => {
            const normalizedExample =
                this.normalizeVectorToUnitLength(tf.reshape(input, [input.size]));
            const exampleSize = normalizedExample.shape[0];

            // Lazily create the logits matrix for all training examples if necessary.
            if (this.trainDatasetMatrix == null) {
                let newTrainLogitsMatrix = null;

                for (const label in this.classDatasetMatrices) {
                    newTrainLogitsMatrix = concatWithNulls(
                        newTrainLogitsMatrix, this.classDatasetMatrices[label]);
                }
                this.trainDatasetMatrix = newTrainLogitsMatrix;
            }

            if (this.trainDatasetMatrix == null) {
                console.warn('Cannot predict without providing training examples.');
                return null;
            }

            tf.keep(this.trainDatasetMatrix);

            const numExamples = this.getNumExamples();
            return tf.reshape(
                tf.matMul(
                    tf.reshape(this.trainDatasetMatrix, [numExamples, exampleSize]),
                    tf.reshape(normalizedExample, [exampleSize, 1])
                ), [numExamples]);
        });
    }

    /**
     * Predicts the class of the provided input using KNN from the previously-
     * added inputs and their classes.
     *
     * @param input The input to predict the class for.
     * @returns A dict of the top class for the input and an array of confidence
     * values for all possible classes.
     */
    async predictClass(input, k = 3) {
        if (k < 1) {
            throw new Error(
                `Please provide a positive integer k value to predictClass.`);
        }
        if (this.getNumExamples() === 0) {
            throw new Error(
                `You have not added any examples to the KNN classifier. ` +
                `Please add examples before calling predictClass.`);
        }
        const knn = tf.tidy(() => tf.cast(this.similarities(input), 'float32'));
        const kVal = Math.min(k, this.getNumExamples());
        const topKIndices = topK(await knn.data(), kVal).indices;
        knn.dispose();

        return this.calculateTopClass(topKIndices, kVal);
    }

    /**
     * Clears the saved examples from the specified class.
     */
    clearClass(label) {
        if (this.classDatasetMatrices[label] == null) {
            throw new Error(`Cannot clear invalid class ${label}`);
        }

        this.classDatasetMatrices[label].dispose();
        delete this.classDatasetMatrices[label];
        delete this.classExampleCount[label];
        this.clearTrainDatasetMatrix();
    }

    clearAllClasses() {
        for (const label in this.classDatasetMatrices) {
            this.clearClass(label);
        }
    }

    getClassExampleCount() {
        return this.classExampleCount;
    }

    getClassifierDataset() {
        return this.classDatasetMatrices;
    }

    getNumClasses() {
        return Object.keys(this.classExampleCount).length;
    }

    setClassifierDataset(classDatasetMatrices) {
        this.clearTrainDatasetMatrix();

        this.classDatasetMatrices = classDatasetMatrices;
        for (const label in classDatasetMatrices) {
            this.classExampleCount[label] = classDatasetMatrices[label].shape[0];
        }
    }

    /**
     * Calculates the top class in knn prediction
     * @param topKIndices The indices of closest K values.
     * @param kVal The value of k for the k-nearest neighbors algorithm.
     */
    calculateTopClass(topKIndices, kVal) {
        let topLabel;
        const confidences = {};

        if (topKIndices == null) {
            // No class predicted
            return {
                classIndex: this.labelToClassId[topLabel],
                label: topLabel,
                confidences
            };
        }

        const classOffsets = {};
        let offset = 0;
        for (const label in this.classDatasetMatrices) {
            offset += this.classExampleCount[label];
            classOffsets[label] = offset;
        }
        const votesPerClass = {};
        for (const label in this.classDatasetMatrices) {
            votesPerClass[label] = 0;
        }
        for (let i = 0; i < topKIndices.length; i++) {
            const index = topKIndices[i];
            for (const label in this.classDatasetMatrices) {
                if (index < classOffsets[label]) {
                    votesPerClass[label]++;
                    break;
                }
            }
        }

        // Compute confidences.
        let topConfidence = 0;
        for (const label in this.classDatasetMatrices) {
            const probability = votesPerClass[label] / kVal;
            if (probability > topConfidence) {
                topConfidence = probability;
                topLabel = label;
            }
            confidences[label] = probability;
        }

        return {
            classIndex: this.labelToClassId[topLabel],
            label: topLabel,
            confidences
        };
    }

    /**
     * Clear the lazily-loaded train logits matrix due to a change in
     * training data.
     */
    clearTrainDatasetMatrix() {
        if (this.trainDatasetMatrix != null) {
            this.trainDatasetMatrix.dispose();
            this.trainDatasetMatrix = null;
        }
    }

    /**
     * Normalize the provided vector to unit length.
     */
    normalizeVectorToUnitLength(vec) {
        return tf.tidy(() => {
            const sqrtSum = tf.norm(vec);

            return tf.div(vec, sqrtSum);
        });
    }

    getNumExamples() {
        let total = 0;
        for (const label in this.classDatasetMatrices) {
            total += this.classExampleCount[label];
        }

        return total;
    }

    dispose() {
        this.clearTrainDatasetMatrix();
        for (const label in this.classDatasetMatrices) {
            this.classDatasetMatrices[label].dispose();
        }
    }
}

export class KNN {

    constructor() {
        this.model = null;
        this.labels = [];
        this.hyperparams = {
            K: 5,
        };
    }

    setHyperParameters(params) {
        this.hyperparams = params;
    }

    train(features, percentageForValidation = 0) {

        this.labels = Array.from(features.keys());
        let tensors = getSuffledAndSplittedTensorsAndValidationData(features, percentageForValidation);       
        
        ///////////// KNN ////////////
        const classifier = new KNNClassifier();
        let inputs = tf.unstack(tensors.inputTensor);
        let outputs = tf.unstack(tensors.outputTensor);

        inputs.forEach((input, index) => {
            console.log(input);
            console.log(outputs[index].arraySync().indexOf(1));
            let label = outputs[index].arraySync().indexOf(1);
            classifier.addExample(input, label);

        });
        //////////////////////////////

        this.model = classifier;
        let info = "NO INFO";

        let result = getFormattedModelResult(this.model, info, tensors.validationData);         

        return Promise.resolve(result);
    }

    classify(inputTensor) {
        let k = this.hyperparams.K;
        return this.model.predictClass(inputTensor, k).then(prediction => {
            let results = [];
            for (let i in prediction.confidences) {
                results.push([this.labels[i], prediction.confidences[i]]);
            }
            results.sort((a, b) => b[1] - a[1]);
            inputTensor.dispose();
            return results;
        })
    }
}