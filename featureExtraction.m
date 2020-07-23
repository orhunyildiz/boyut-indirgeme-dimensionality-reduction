function [performanceknn, sayac, train, test] = featureExtraction(bestSolution, train, test, k, r)

[~, columnIndex] = size(bestSolution);
sayac = 0;
i = columnIndex;
% for i=columnIndex:1
%     if(bestSolution(1,i) < r)
%             train(:,i + 1) = [];
%             test(:,i + 1) = [];
%             sayac = sayac + 1;
%     end
% end

while i > 0
    if(bestSolution(1,i) < r)
        train(:,i + 1) = [];
        test(:,i + 1) = [];
        sayac = sayac + 1;
    end
    i = i - 1;
end

[performanceknn] = knn(train, test, k);

end
