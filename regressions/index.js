require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression-tf');

const {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
});

const lr = LinearRegression(features, labels, { learningRate: 0.000008, iterations: 1000 });
const model = lr.train();
console.log('model', model);
model.predict(130);