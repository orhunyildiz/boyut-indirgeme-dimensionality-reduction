function [ class ] = findClass( label, k )

    labels = unique(label);
    cLabels = zeros(1, numel(labels));
    for i = 1 : k
        index = find(labels == label(i));
        cLabels(index) = cLabels(index) + 1;
    end
    [~, mI] = max(cLabels);
    class = labels(mI);
end