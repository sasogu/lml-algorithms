import { getMobilenetEncoder } from "./feature-extraction/mobilenet";
import { numericalEncoder } from "./feature-extraction/numerical";
import { useEncoder } from "./feature-extraction/useEncoder";

import { LMLSequential } from "./algorithms/sequential";
import { KNN } from "./algorithms/knn";
import { confusionMatrix } from "./algorithms/util";

function saveDatasetInLocalStorage(clave, dataset) {
        const serializableMap = {};
        dataset.forEach((value, key) => {
                // Convertir el Set a un array antes de serializar
                serializableMap[key] = Array.from(value);
        });
        const serializado = JSON.stringify(serializableMap);
        localStorage.setItem(clave, serializado);
}

function getDatasetFromLocalStorage(clave) {
        const serializado = localStorage.getItem(clave);
        if (serializado) {
                const serializableMap = JSON.parse(serializado);
                const mapConSet = new Map();
                Object.keys(serializableMap).forEach((key) => {
                        // Convertir el array de vuelta a un Set
                        mapConSet.set(key, new Set(serializableMap[key]));
                });
                return mapConSet;
        }
        return new Map();
}

export {
        getMobilenetEncoder,
        numericalEncoder,
        useEncoder,
        LMLSequential,
        KNN,
        confusionMatrix,
        saveDatasetInLocalStorage,
        getDatasetFromLocalStorage
}