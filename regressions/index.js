require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression-tf');

const {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const lr = LinearRegression(features, labels, { learningRate: 0.1, iterations: 100 });
const model = lr.train();

// print weights
console.log('weights:');
model.weights.print();

// print accuracy (only for tf)
console.log('accuracy:');
const test = model.test(testFeatures, testLabels);
test.result.print();
test.r2.print();

const values = [130, 1.75, 307];
console.log('prediction:', values);
const prediction = model.predict(...values);
prediction.print();