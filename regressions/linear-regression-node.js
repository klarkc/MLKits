function buildGradientDescenter(features, labels, options) {
    return ({m, b}) => {
        // guess is result of f(x) = mx + b, x is the feature
        const currentFeatureGuesses = features
        .map(([feature]) => {
            // console.log(`m(${m}) * x(${feature}) + b(${b})`);
            return m * feature + b;
        });
        // console.log(currentFeatureGuesses);
        
        // slope of MSE in respect to B
        // 2/n * SUM(guess - actual) | n is number of rows
        const bSlope = currentFeatureGuesses
            .map((guess, index) => {
                const actual = labels[index][0];
                // console.log('guess', guess, 'actual', actual);
                return guess - actual;
            })
            .reduce((sum, res) => sum + res, 0) * (2 / features.length);
        // slope of MSE in respect to M
        // 2/n * SUM(-x * [actual - guess]) | n is number of rows, x is feature at row
        const mSlope = currentFeatureGuesses
            .map((guess, index) => {
                const actual = labels[index][0];
                // console.log('x', features[index][0], 'guess', guess, 'actual', actual);
                return -1 * features[index][0] * ( actual - guess);
            })
            .reduce((sum, res) => sum + res, 0) * (2 / features.length)
            // console.log('m', m, mSlope, 'b', b, bSlope);
            return {
                m: m - mSlope * options.learningRate,
                b: b - bSlope * options.learningRate
            }
    }
}

function buildTrainer(gradientDescenter, buildModel, options) {
    const {m, b} = Array(options.iterations)
        .fill(null)
        .reduce(
            prev => gradientDescenter(prev),
            { m: 0, b: 0 }
        );
    return () => buildModel(m, b);
}

function buildModel(m, b) {
    return {
        m,
        b,
        predict(feature) {
            const result = m * feature + b;
            console.log('result', result);
        }
    }
}

module.exports = function LinearRegression(features, labels, options = {}) {
    const defOptions = { learningRate: 0.1, iterations: 1000};
    const myOptions = {...defOptions, ...options};
    const gradientDescenter = buildGradientDescenter(features, labels, myOptions);
    const train = buildTrainer(gradientDescenter, buildModel, myOptions);

    return {
        gradientDescenter,
        train,
    }
}