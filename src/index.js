import { getMobilenetEncoder } from "./feature-extraction/mobilenet";
import { numericalEncoder } from "./feature-extraction/numerical";
import { useEncoder } from "./feature-extraction/useEncoder";

import { LMLSequential } from "./algorithms/sequential";
import { KNN } from "./algorithms/knn";
import { LMLModelFactory } from "./algorithms/modelFactory";
import { confusionMatrix } from "./algorithms/util";

function serializeMap(m) {
        const serializableMap = {};
        m.forEach((value, key) => {
                // Convertir el Set a un array antes de serializar
                serializableMap[key] = Array.from(value);
        });
         
        return JSON.stringify(serializableMap);
}

function deserializeToMap(s) {
        const serializado = s;
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
        LMLModelFactory,
        confusionMatrix,
        serializeMap,
        deserializeToMap
}