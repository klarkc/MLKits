const tf = require('@tensorflow/tfjs');

function buildGradientDescenter(features, labels, options) {
    const tFeatures = tf.tensor(features);
    const tLabels = tf.tensor(labels);
    const oneFeatures = tf.ones([tFeatures.shape[0], 1]).concat(tFeatures, 1);
    return (weights) => {
        // slope of MSE with respect to m and b
        // (Features * ((Features - Weights) - Labels) / n
        const currentGuesses = oneFeatures.matMul(weights);
        const differences = currentGuesses.sub(tLabels);
        const slopes = oneFeatures
            .transpose()
            .matMul(differences)
            .div(oneFeatures.shape[0]);
        const lrSlopes = slopes.mul(options.learningRate);
        return weights.sub(lrSlopes);
    }
}

function buildTrainer(gradientDescenter, buildModel, options) {
    let zeroWeights = tf.zeros([2, 1]);   
    return () => {
        let weights = zeroWeights;
        // find ideal b, m
        for (let i = options.iterations; i > 0; i--) {
            weights = gradientDescenter(weights);       
        }
        return buildModel(weights);
    }
}

function buildModel(weights) {
    return {
        weights,
        async predict(feature) {
            const [b, m] = await weights.data();
            const result = m * feature + b;
            return result;
        }
    }
}

module.exports = function LinearRegression(features, labels, options = {}) {
    const defOptions = { learningRate: 0.1, iterations: 1000 };
    const myOptions = { ...defOptions, ...options };
    const gradientDescenter = buildGradientDescenter(features, labels, myOptions);
    const train = buildTrainer(gradientDescenter, buildModel, myOptions);

    return {
        gradientDescenter,
        train,
    }
}