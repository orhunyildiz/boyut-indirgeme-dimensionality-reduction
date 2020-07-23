function [bestSolution, bestFitness, iteration]=SFS_0004(train, test, maxIteration, k)

settings;
[~, dimension] = size(train);
S.Start_Point = 50;            
S.Maximum_Diffusion = 0;
S.Walk = 1; % *Important
S.Ndim = dimension - 1;
S.Lband = zeros(1, S.Ndim);
S.Uband = ones(1, S.Ndim);
S.Maximum_Generation  = maxIteration;
    P = zeros(S.Start_Point,S.Ndim);
%Creating random points in considered search space=========================
    point = repmat(S.Lband,S.Start_Point,1) + rand(S.Start_Point, S.Ndim).* ...
        (repmat(S.Uband - S.Lband,S.Start_Point,1));
%==========================================================================

%Calculating the fitness of first created points=========================== 
    FirstFit = zeros(1,S.Start_Point);
    for i = 1 : size(point,1)
        FirstFit(i) = testFunction(point(i,:)', train, test, k);
    end
    [Sorted_FitVector, Indecis] = sort(FirstFit);
    point = point(Indecis,:);%sorting the points based on obtaind result
%==========================================================================
    
%Finding the Best point in the group=======================================
    BestPoint = point(1, :);
    fbest = Sorted_FitVector(1);%saving the first best fitness
%==========================================================================
nfeval = 1;
%Starting Optimizer========================================================
while ( ( nfeval < S.Maximum_Generation) )
    New_Point = point;
    FitVector = Sorted_FitVector;
    %diffusion process occurs for all points in the group
    if S.Maximum_Diffusion>0
        for i = 1 : S.Start_Point
            %creating new points based on diffusion process
            [NP, fit] = Diffusion_Process(point(i,:),Sorted_FitVector(i),S,nfeval,BestPoint,fhd, fNumber);
            New_Point(i,:) = NP; FitVector(i) = fit;
            nfeval = nfeval + 1;
           if nfeval >= S.Maximum_Generation
               S.Start_Point = 0;
               break;
           end
        end  
    end
    fit = FitVector';
    [~, sortIndex] = sort(fit);
    
    Pa = zeros(1,S.Start_Point);
    %Starting The First Updating Process====================================
    for i=1:1:S.Start_Point     
        Pa(sortIndex(i)) = (S.Start_Point - i + 1) / S.Start_Point; 
    end

    RandVec1 = randperm(S.Start_Point);
    RandVec2 = randperm(S.Start_Point);
    

    for i = 1 : S.Start_Point
        FDBIndex = FDBDinamik( point, fit, 100, nfeval,S.Maximum_Generation );
        for j = 1 : size(New_Point,2)
            if rand > Pa(i)
                if rand < 0.5
                    P(i,j) = New_Point(FDBIndex,j) - rand*(New_Point(RandVec2(i),j) - New_Point(i,j)); 
                else
                    P(i,j) = New_Point(RandVec1(i),j) - rand*(New_Point(RandVec2(i),j) - New_Point(i,j)); 
                end
            else
                P(i,j)= New_Point(i,j);
            end
        end
    end
    P = Bound_Checking(P,S.Lband,S.Uband);%for checking bounds
    for i = 1 : S.Start_Point
        Fit_FirstProcess = testFunction(P(i,:)', train, test, k);
        if Fit_FirstProcess<=fit(i)
            New_Point(i,:)=P(i,:);
            fit(i)=Fit_FirstProcess;
        end
        nfeval = nfeval + 1;
       if nfeval >= S.Maximum_Generation
           S.Start_Point = 0;
           break;
       end
    end

    FitVector = fit;
    %======================================================================    
       
	[Sorted_FitVector,SortedIndex] = sort(FitVector);
    New_Point = New_Point(SortedIndex,:);
    BestPoint = New_Point(1,:);%first point is the best  
    
    
    pbest = New_Point(1,:);
    fbest = FitVector(1);
    point = New_Point;
    
    %Starting The Second Updating Process==================================
    Pa = sort(SortedIndex/S.Start_Point, 'descend');
    
    for i = 1 : S.Start_Point
       if rand > Pa(i)
           %selecting two different points in the group
           R1 = ceil(rand*size(point,1));
           R2 = ceil(rand*size(point,1));
            while R1 == R2
                R2 = ceil(rand*size(point,1));
            end
            
            if rand < .5
                ReplacePoint = point(i,:) - rand * (point(R2,:) - BestPoint); 
            else
                ReplacePoint = point(i,:) + rand * (point(R2,:) - point(R1,:)); 
            end
            ReplacePoint = Bound_Checking(ReplacePoint,S.Lband,S.Uband);
            replaceFit = testFunction(ReplacePoint', train, test, k);
            if replaceFit < Sorted_FitVector(i)
                point(i,:) = ReplacePoint;
                Sorted_FitVector(i) = replaceFit;
            end
            if replaceFit < fbest
                pbest = ReplacePoint;
                fbest = replaceFit;
                BestPoint = pbest;
            end
            nfeval = nfeval + 1;
            if nfeval >= S.Maximum_Generation
               break;
            end
       end
    end
end
bestFitness=fbest;
bestSolution=pbest;
iteration = nfeval;
end

function p = Bound_Checking(p,lowB,upB)
    for i = 1 : size(p,1)
        upper = double(gt(p(i,:),upB));
        lower = double(lt(p(i,:),lowB));
        up = find(upper == 1);
        lo = find(lower == 1);
        if (size(up,2)+ size(lo,2) > 0 )
            for j = 1 : size(up,2)
                p(i, up(j)) = (upB(up(j)) - lowB(up(j)))*rand()...
                    + lowB(up(j));
            end
            for j = 1 : size(lo,2)
                p(i, lo(j)) = (upB(lo(j)) - lowB(lo(j)))*rand()...
                    + lowB(lo(j));
            end
        end
    end
end

function [createPoint, fitness] = Diffusion_Process(Point,Fitness,S,g,BestPoint, fhd, fNumber)
    %calculating the maximum diffusion for each point
    NumDiffiusion = S.Maximum_Diffusion;
    New_Point = zeros(S.Maximum_Diffusion+1,S.Ndim);
    fitness = zeros(1,S.Maximum_Diffusion+1);
    New_Point(1,:) = Point;
    fitness(1) = Fitness;
    %Diffiusing Part*******************************************************
    for i = 1 : NumDiffiusion
        %consider which walks should be selected.
        if rand < S.Walk 
            GeneratePoint = normrnd(BestPoint, (log(g)/g)*(abs((Point - BestPoint))), [1 size(Point,2)]) + (randn*BestPoint - randn*Point); % Eşitlik (11)
        else
            GeneratePoint = normrnd(Point, (log(g)/g)*(abs((Point - BestPoint))),[1 size(Point,2)]); % Eşitlik (12) 
        end
        New_Point(i+1,:) = GeneratePoint;
    end
    %check bounds of New Point
    New_Point = Bound_Checking(New_Point,S.Lband,S.Uband);
    %sorting fitness
    for i = 2 : size(New_Point,1)
        fitness(i) = testFunction(New_Point(i,:)', train, test, k);
    end

    [fit_value,fit_index] = sort(fitness);
    fitness = fit_value(1,1);
    New_Point = New_Point(fit_index,:);
    createPoint = New_Point(1,:);
    %======================================================================
end

