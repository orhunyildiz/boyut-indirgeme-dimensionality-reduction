function [ecosize, n, maxFE, lb, ub] = problem_terminate(train)

    % Parameter settings:
     
    % ecosystem (population) size
    ecosize = 30;

    % problem 
    [~, column] = size(train);
    n = column - 1;

    % number of function evaluations
    maxFE = 50 * n;

    % problem lower band 
    lowerBand = 0;
    lb = ones(1, n) * lowerBand;

    % problem upper band
    upperBand = 1;
    ub = ones(1, n) * upperBand;

end

