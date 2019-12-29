const tf = require('@tensorflow/tfjs');

function buildGradientDescenter(features, labels, options) {
    return ({m, b}) => {
        // guess is result of f(x) = mx + b, x is the feature
        const currentFeatureGuesses = features
            .map(([feature]) => m * feature + b);

        // slope of MSE in respect to B
        // 2/n * SUM(guess - actual) | n is number of rows
        const bSlope = currentFeatureGuesses
            .map((guess, index) => {
                const actual = labels[index][0];
                return guess - actual;
            })
            .reduce((sum, res) => sum + res, 0) * (2 / features.length);
        // slope of MSE in respect to M
        // 2/n * SUM(-x * [actual - guess]) | n is number of rows, x is feature at row
        const mSlope = currentFeatureGuesses
            .map((guess, index) => {
                const actual = labels[index][0];
                return -1 * features[index][0] * ( actual - guess)
            })
            .reduce((sum, res) => sum + res, 0) * (2 / features.length)

            return {
                m: m - mSlope * options.learningRate,
                b: b - bSlope * options.learningRate
            }
    }
}

function buildTrainer(gradientDescenter, options) {
    return () => Array(options.iterations).fill({ m: 0, b: 0 }).map(gradientDescenter);
}

module.exports = function LinearRegression(features, labels, options = {}) {
    const defOptions = { learningRate: 0.1, iterations: 1000};
    const myOptions = {...defOptions, options};

    const gradientDescenter = buildGradientDescent(features, labels, myOptions);
    const trainer = buildTrainer(gradientDescenter, myOptions);

    return {
        gradientDescenter,
        trainer,
    }
}