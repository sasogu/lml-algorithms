import { LMLSequential } from './sequential';
import { KNN } from './knn';

export class LMLModelFactory {

    static createModel(type) {

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
     * 1) si se ha cargado un archivo de modelo desde 
     *    la GUI de Scratch, es decir, existe `modelZipContent`, entonces se
     *    crea un modelo con las característica del contenido de modelZipContent.
     * 2) si en el localStorage hay un item lmlModelMetadata,
     *    se crea un modelo con las características del contenido de ese item,
     * 3) en otro caso, se crea un nuevo modelo por defecto (sequential)
     * 
     * @param {*} mode 
     * @param {*} modelZipContent 
     * @returns 
     */
    static load(modelZipContent = null) {

        let metadata;

        if (modelZipContent != null) {
            let name = Object.keys(modelZipContent.files)[0].split(".")[0];
            metadata = modelZipContent.files[name + '.metadata'];
            return metadata.async("string").then(content => {
                let lmlModelMetadata = JSON.parse(content);
                let model = LMLModelFactory.createModel(lmlModelMetadata.modelAlgorithm);
                return model.loadFromZipContent(modelZipContent);
            })
        } else if (localStorage.getItem('lmlModel') != null) {
            let lmlModel = JSON.parse(localStorage.getItem('lmlModel'));
            let name = lmlModel.data.name;            
            let model = LMLModelFactory.createModel(lmlModel.model.metadata.modelAlgorithm);
            return model.loadFromLocalstorage();

        } else {
            let model = LMLModelFactory.createModel('sequential')
            let result = {
                lmlModel: model,
                lmlModelMetadata: {
                    model: {
                        jsonFile: null,
                        weights: null,
                        metadata: {
                            modelAlgorithm: 'sequential',
                            hyperparameters: {
                                learningRate: 0.001,
                                batchSize: 10,
                                epochs: 20
                            },
                            labels: []
                        }

                    },
                    data: {
                        type: 'text',
                        name: 'undefined'
                    }
                },
                loadedFrom: 'blank'
            }

            return Promise.resolve(result);
        }

    }
}