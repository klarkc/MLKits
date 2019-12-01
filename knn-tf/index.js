require('@tensorflow/tfjs-node');
const yargs = require('yargs');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('./load-csv');

function prepare(dataColumns = ['lat', 'long'], numberOfTests = 10) {
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
    
    function standardizate(tensor) {
        const {mean, variance} = tf.moments(tensor, 0);
        return tensor
            .sub(mean)
            .div(variance.pow(.5));
    }

    function knn(testPoint, k) {
        // d = (d1² + d2²) ** 0.5
        // d = (fLat - lLat - fLon - lLon)² + ... ** 0.5
        // console.log('knn', testPoint.arraySync(), k);
        const stded = standardizate(features.concat(testPoint.expandDims()));
        const stFeats = stded.slice([0,0], [stded.shape[0] - 1, -1]);
        const stTestPoint = stded.slice([stded.shape[0] - 1, 0], [1, -1]);
        return stFeats
            .sub(stTestPoint)
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
        console.log('Running accuracy tests...', dataColumns, 'features', numberOfTests, 'tests', numberOfKs, 'K\'s');
        const kArray = [...Array(numberOfKs)].fill(null);
        const tArray = [...Array(numberOfTests)].fill(null);
        const tests = tf.tensor(tArray.map((_, tIdx) =>
            kArray.map((_, kIdx) => knn(testFeatures.slice([tIdx, 0], [1, -1]).reshape([-1]), rBegin + kIdx))
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
            .positional('kBegin', {type: 'number', default: 1})
            .positional('kEnd', { type: 'number', default: 20 })
            .option('tests', { type: 'number', default: 10 })
            .option('features', { type: 'array', default: ['lat', 'long'] })
        prepare(yargs.argv.features, yargs.argv.tests).
            accuracyOfKs([yargs.argv.kBegin, yargs.argv.kEnd]);
    })
    .command('predict', 'predicts with inputs', (yargs) => {
        yargs
        .option('k', { type: 'number', default: 10 })
        .option('features', { type: 'array', default: ['lat' , 'long']})
        .option('inputs', {type: 'array', default: [47.4231, -122.200]})
    prepare(yargs.argv.features)
            .predict(tf.tensor(yargs.argv.inputs), yargs.argv.k);
    })
    .demandCommand()
    .help()
    .argv;