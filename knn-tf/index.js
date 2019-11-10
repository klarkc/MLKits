require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('./load-csv');
const splitTest = 10;
const csv = loadCsv('kc_house_data.csv', {
    shuffle: true,
    splitTest,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price'],
});
const features = tf.tensor(csv.features);
const labels = tf.tensor(csv.labels);
const testFeatures = tf.tensor(csv.testFeatures);
const testLabels = tf.tensor(csv.testLabels);

function knn(testPoint, k) {
    // d = (d1² + d2²) ** 0.5
    // d = (fLat - lLat - fLon - lLon)² + ... ** 0.5
    return features
        .sub(testPoint)
        .pow(2)
        .sum(1) // sum d1², d2²
        .pow(.5) // ** 0.5
        .expandDims(1)
        .concat(labels, 1)
        .unstack() // array of tensors
        .sort((t1, t2) => {
            const t1d = t1.dataSync();
            const t2d = t2.dataSync();
            return t1d[0] - t2d[0]; // by distance
        })
        .slice(0, k)
        // nearist avarage label values
        .reduce((sum, tensor) => sum + tensor.dataSync()[1], 0) / k; 
}

function accuracyOfKs(range = [1, 20]) {
    console.log('Accuracy of Ks...');
    const results = [...Array(range[1]).keys()].map(
        k => testFeatures.dataSync().filter(
            testPoint => knn(testPoint, k) == // TODO
        )
    );
    console.table(results);
}

function predict(input) {
    const prediction = knn(input, 10) ;
    console.log('prediction', prediction);
}

accuracyOfKs();
// predict(tf.tensor([47.4231, -122.200]));