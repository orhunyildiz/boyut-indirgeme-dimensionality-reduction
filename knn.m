function [error] = knn(train, test, kn)
    [testCount, column] = size(test);
    [trainCount, ~] = size(train);
    distance = zeros(testCount, trainCount);
    
    for i = 1 : testCount
        for j = 1 : trainCount
            for k = 2 : column
                distance(i, j) = distance(i, j) + (train(j, k) - test(i, k))^2;
            end
        end
    end
    
    %en kucuk elemanlar ve indeksleri 
    sortDistance = zeros(testCount, trainCount);
    sortIndex = zeros(testCount, trainCount);
    testLabel = zeros(1, testCount);
    trueCount = 0; falseCount = 0;
    for i = 1 : testCount
        [sorted, index] = sort(distance(i, :));
        sortDistance(i, :) = sorted;
        sortIndex(i, :) = index;
        testLabel(i) = findClass(train(index, 1), kn);
        
        if(testLabel(i) == test(i, 1))
            trueCount = trueCount + 1;
        else, falseCount = falseCount + 1;
        end
    end
    
    error = (100 * falseCount) / testCount ; 
end