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

const lr = LinearRegression(features, labels, { learningRate: 0.1, iterations: 100 });
const model = lr.train();

// print weights
console.log('weights:');
model.weights.print();

// print accuracy (only for tf)
console.log('accuracy:');
model.test(testFeatures, testLabels).print();

const value = 130;
console.log('prediction:', value);
model.predict(value).then(
    res =>console.log('result:', res)
);