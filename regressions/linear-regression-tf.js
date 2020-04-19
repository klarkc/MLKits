const tf = require('@tensorflow/tfjs');

function prependOnes(tensor) {
    return tf.ones([tensor.shape[0], 1]).concat(tensor, 1);
}

function buildStandardnizer(baseFeatures) {
    const tBaseFeatures = tf.tensor(baseFeatures);
    const { mean, variance } = tf.moments(tBaseFeatures);
    return features => features.sub(mean).div(variance.pow(0.5));
}

function buildGradientDescenter(features, labels, standard, options) {
    const tFeatures = standard(tf.tensor(features));
    const tLabels = tf.tensor(labels);
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

function createTesterBuilder(standard) {
    // Coefficient of Determination formula
    // R ² = 1 - SSres / SStot
    // SS = Sum of Squares
    // SSres = sum((Actual - Predicted)²)
    // SStot = sum((Actual - Avarage)²)
    return (weights) => (testFeatures, testLabels) => {
        const tFeatures = standard(tf.tensor(testFeatures));
        const oneFeatures = prependOnes(tFeatures);
        const tLabels = tf.tensor(testLabels);
        const predictions = oneFeatures.matMul(weights);
        tf.tensor(testFeatures)
            .concat(predictions, 1)
            .concat(tLabels, 1)
            .print();
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

function createModelBuilder(buildTester, standard) {
    return weights => {
        const model = {
            weights,
            test: buildTester(weights),
            async predict(...features) {
                const tFeatures = standard(tf.tensor(features));
                const [b, m] = await weights.data();
                const [feature] = await tFeatures.data();
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
    const standard = buildStandardnizer(features);
    const gradientDescenter = buildGradientDescenter(features, labels, standard, myOptions);
    const testerBuilder = createTesterBuilder(standard);
    const modelBuilder = createModelBuilder(testerBuilder, standard);
    const train = buildTrainer(gradientDescenter, modelBuilder, myOptions);

    return {
        gradientDescenter,
        train,
    }
}