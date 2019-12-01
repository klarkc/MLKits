require('@tensorflow/tfjs-node');
const yargs = require('yargs');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('./load-csv');

function prepare(numberOfTests = 10, dataColumns = ['lat', 'long']) {
    const csv = loadCsv('kc_house_data.csv', {
        shuffle: true,
        splitTest: numberOfTests,
        dataColumns,
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

    function accuracyOfKs([rBegin, rEnd] = [1, 20]) {
        const numberOfKs = rEnd + 1 - rBegin;
        console.log('Running accuracy tests...', numberOfTests, 'tests', numberOfKs, 'K\'s');
        const kArray = [...Array(numberOfKs)].fill(null);
        const tArray = [...Array(numberOfTests)].fill(null);
        const tests = tf.tensor(tArray.map((_, tIdx) =>
            kArray.map((_, kIdx) => knn(testFeatures.slice([tIdx, 0], [1, -1]), rBegin + kIdx))
        ));
        const results = tf.stack(
            tests
            // cur shape: [tests, k's]
            .unstack(1)
            .map(testPredictions =>  tf.metrics.meanAbsolutePercentageError(testLabels.reshape([-1]), testPredictions))
        )
            .unstack()
            // generate [k, mseSum]
            .map((mseSum, kIdx) => ([kIdx + rBegin, mseSum.arraySync()]))
            // sort from smaller to biggest
            .sort((prev, next) => prev[1] - next[1]);
        console.log('from more accurate to less accurate');
        console.table(results);
        return results;
    }

    function predict(input, k = 10) {
        const prediction = knn(input, k);
        console.log('prediction of', input.arraySync(), 'with k', k, 'is', prediction);
    }

    return {knn, accuracyOfKs, predict}
}

yargs
    .command(['accuracy [kBegin] [kEnd]'], 'show accuracy', (yargs) => {
        yargs
            .positional('kBegin', {type: Number, default: 1})
            .positional('kEnd', { type: Number, default: 20 })
            .option('tests', { type: Number, default: 10 })
            .option('features', { type: Array, default: ['lat', 'long'] })
        prepare(yargs.argv.tests, yargs.argv.features).
            accuracyOfKs([yargs.argv.kBegin, yargs.argv.kEnd]);
    })
    .command('predict [features..]', 'predicts', (yargs) => {
        yargs
        .option('k', { type: Number, default: 10 })
        .positional('features', { type: Array, default: [47.4231, -122.200] })
        prepare()
            .predict(tf.tensor(yargs.argv.features), yargs.argv.k);
    })
    .demandCommand()
    .help()
    .argv;