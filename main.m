
[train, test] = readData('diabetic');

k = 5;
maxIteration = 300;
threshold = 0.3;
%düzeltilenler= agde, fdb_agde, aeo, aso, 
%[errorKnn] = knn(train, test, k);
%[weights, errorAGDE, iter] = agde(train, test, maxIteration, k);
%weights = transpose(weights);

[weights, performance, iter] = Mfro_007(train, test, maxIteration, k);
[performanceKNN, counter, train, test] = featureExtraction(weights, train, test, k, threshold);
