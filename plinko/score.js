const outputs = [];

function distance(pointA, pointB) {
  return Math.sqrt(
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => Math.pow(Math.abs(a - b), 2))
      .sum()
      .value()
  );
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const traningSet = _.slice(shuffled, testCount);
  return [testSet, traningSet];
}

function knn(traningSet, testPoint, k) {
    // trainingSet row is [ballPos, ballBounce, ballSize, bucket]
    // testPoint is [ballPos, ballBounce, ballSize]
    return _.chain(traningSet)
    .map(row => [
      distance(_.initial(row), testPoint),
      _.last(row) // bucket
    ])
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function minMax(data, featureCount) {
  const columns = _.chain(_.range(0, featureCount))
    .map((_, idx) => data.map(
      dataValue => dataValue[idx]
    ))
    .value();
  const normalizer = (value, colIdx) => {
    const column = columns[colIdx];
    const min = _.min(column);
    const max = _.max(column);
    const res = (value - min) / (max - min);
    if (Number.isNaN(res)) return 1; // NaN means that all values are equal
    return res;
  }
  return _.chain(data)
    .map(row => row.slice(0, featureCount))
    .map(row => row.map(normalizer))
    .map((row, idx) => ([
        ...row,
        ...data[idx].slice(featureCount)
    ]))
    .value();
}

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function accuracyOfKs({kRange = [1, 20], featureCount = 3, testSetSize = 100} = {}) {
  console.log('Accuracy of Ks...');
  const normOutputs = minMax(outputs, featureCount);
  const [testSet, traningSet] = splitDataset(normOutputs, testSetSize);
  const table = _.range(...kRange).map(k => {
    return _.chain(testSet)
      .filter(
        testPoint => knn(traningSet, _.initial(testPoint), k) === _.last(testPoint)
      )
      .size()
      .divide(testSetSize)
      .value();
    });
  console.table(table);
}

function accuracyOfFeats({featureRange = [0, 3], k = 10, testSetSize = 100} = {}) {
  console.log('Accuracy of features...');
  const table = _.range(...featureRange).map(feat => {
    const outputsOfFeat = outputs.map(o => [o[feat], _.last(o)]);
    const normOutputs = minMax(outputsOfFeat, 1);
    const [testSet, traningSet] = splitDataset(normOutputs, testSetSize);
    return _.chain(testSet)
      .filter(
        testPoint => knn(traningSet, _.initial(testPoint), k) === _.last(testPoint)
      )
      .size()
      .divide(testSetSize)
      .value();
  });
  console.table(table);
}

function predict({point = [300, .55, 16], featureCount = 3, k = 10} = {}) {
  const traningSet = minMax(outputs, featureCount);
  const prediction = knn(traningSet, point, k);
  console.log('bucket prediction', prediction);
}

function runAnalysis() {
  // accuracyOfKs({featureRange: [0, 1]});
  // accuracyOfFeats({k: 10});
  predict({featureCount: 1, k: 8});
}

