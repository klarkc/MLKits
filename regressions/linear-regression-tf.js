const tf = require('@tensorflow/tfjs');

function prependOnes(tensor) {
    return tf.ones([tensor.shape[0], 1]).concat(tensor, 1);
}

function buildGradientDescenter(features, labels, options) {
    const tFeatures = tf.tensor(features);
    const tLabels = tf.tensor(labels);
    const oneFeatures = prependOnes(tFeatures);
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
        weights.isNaN().any().data().then(([hasNan]) => {
            if (hasNan) console.warn('Infinity weight detected, try lowering learning rate');
        });
        return buildModel(weights);
    }
}

function buildTester(testFeatures, testLabels) {
    // Coefficient of Determination formula
    // R ² = 1 - SSres / SStot
    // SS = Sum of Squares
    // SSres = sum((Actual - Predicted)²)
    // SStot = sum((Actual - Avarage)²)
    const tFeatures = tf.tensor(testFeatures);
    const oneFeatures = prependOnes(tFeatures);
    const tLabels = tf.tensor(testLabels);
    return (weights) => {
        const predictions = oneFeatures.matMul(weights);
        const ssRes = tLabels
            .sub(predictions)
            .pow(2)
            .sum();
        const avarage = tLabels.mean();
        const ssTot = tFeatures
            .sub(avarage)
            .pow(2)
            .sum();
        const r2 = tf.tensor([1]).sub(ssRes.div(ssTot));
        return r2;
    }
}

function createModelBuilder(test) {
    return weights => {
        const model = {
            weights,
            accuracy: test(weights),
            async predict(feature) {
                const [b, m] = await weights.data();
                const result = m * feature + b;
                return result;
            },
        };

        return model;
    };
}

module.exports = function LinearRegression(features, labels, options = {}) {
    const defOptions = { learningRate: 0.1, iterations: 1000 };
    const myOptions = { ...defOptions, ...options };
    const gradientDescenter = buildGradientDescenter(features, labels, myOptions);
    const tester = buildTester(features, labels);
    const buildModel = createModelBuilder(tester);
    const train = buildTrainer(gradientDescenter, buildModel, myOptions);

    return {
        gradientDescenter,
        train,
    }
}