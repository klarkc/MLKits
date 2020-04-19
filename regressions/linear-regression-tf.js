const tf = require('@tensorflow/tfjs');

function prependOnes(tensor) {
    return tf.ones([tensor.shape[0], 1]).concat(tensor, 1);
}

function buildStandardnizer(baseFeatures) {
    const tBaseFeatures = tf.tensor(baseFeatures);
    const { mean, variance } = tf.moments(tBaseFeatures);
    return features => features.sub(mean).div(variance.pow(0.5));
}

function buildGradientDescenter(tFeatures, tLabels, options) {
    const oneFeatures = prependOnes(tFeatures);
    return (weights) => {
        // slope of MSE with respect to m and b
        // (Features * (Features * Weights - Labels)) / n
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

function buildTrainer(featuresCount, gradientDescenter, buildModel, options) {
    let zeroWeights = tf.zeros([featuresCount + 1, 1]);   
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

function createTesterBuilder(standard) {
    // Coefficient of Determination formula
    // R ² = 1 - SSres / SStot
    // SS = Sum of Squares
    // SSres = sum((Actual - Predicted)²)
    // SStot = sum((Actual - Avarage)²)
    return (weights) => (testFeatures, testLabels) => {
        const tFeatures = tf.tensor(testFeatures);
        const stFeatures = standard(tFeatures);
        const oneFeatures = prependOnes(stFeatures);
        const tLabels = tf.tensor(testLabels);
        const predictions = oneFeatures.matMul(weights);
        const ssRes = tLabels
            .sub(predictions)
            .pow(2)
            .sum();
        const avarage = tLabels.mean();
        const ssTot = stFeatures
            .sub(avarage)
            .pow(2)
            .sum();
        const r2 = tf.tensor([1]).sub(ssRes.div(ssTot));
        const result = tFeatures
                            .concat(predictions, 1)
                            .concat(tLabels, 1);
        return { r2, result };
    }
}

function createModelBuilder(buildTester, standard) {
    return weights => {
        const model = {
            weights,
            test: buildTester(weights),
            predict(...features) {
                // const [b, m] = await weights.data();
                // const [feature] = await tFeatures.data();
                // const result = m * feature + b;
                const tFeatures = tf.tensor(features).expandDims();
                const stFeatures = standard(tFeatures);
                const oneFeatures = prependOnes(stFeatures);
                const predictions = oneFeatures.matMul(weights);
                return predictions.sum();
            },
        };

        return model;
    };
}

module.exports = function LinearRegression(features, labels, options = {}) {
    const defOptions = { learningRate: 0.1, iterations: 1000 };
    const myOptions = { ...defOptions, ...options };
    const standard = buildStandardnizer(features);
    const tFeatures = standard(tf.tensor(features));
    const tLabels = tf.tensor(labels);

    const gradientDescenter = buildGradientDescenter(tFeatures, tLabels, myOptions);
    const testerBuilder = createTesterBuilder(standard);
    const modelBuilder = createModelBuilder(testerBuilder, standard);
    const train = buildTrainer(tFeatures.shape[1], gradientDescenter, modelBuilder, myOptions);

    return {
        gradientDescenter,
        train,
    }
}