function [bestSolution, bestFitness, iter]=agde(train, test, maxIteration, k)

settings;
[~, dimension] = size(train);
% *************************** %
% ** ALGORITHM�S VARIABLES ** %
% *************************** %
NP = 100;
GEN = ceil(maxIteration/NP);
L = 0;
H = 1;
D = dimension - 1;
X = zeros(D,1); % trial vector
Pop = zeros(D,NP); % population
Fit = zeros(1,NP); % fitness of the population
r = zeros(3,1); % randomly selected indices
% *********************** %
% ** CREATE POPULATION ** %
% *********************** %

for j = 1:NP % initialize each individual
    Pop(:,j) = L + (H-L).*rand(1,D); % within b.constraints
    Fit(1,j)=testFunction(Pop(:,j), train, test, k); 
end
[~, iBest]=min(Fit);
% ****************** %
% ** OPTIMIZATION ** %
% ****************** %

Cr_All = zeros(1,2);
NW = zeros(1,2);

for g = 1:GEN % for each generation
    CrPriods_Index = zeros(1,NP);
    Sr = zeros(1,2);
    CrPriods_Count = zeros(1,2);
    for j = 1:NP % for each individual
         %%%%%%%%ADAPTIVE CR RULE  %%%%%%%%%%%%%%%%%%%%%%%%%
            Ali = rand;
            if(g<=1) % Do for the first Generation
                if (Ali<=1/2)
                    CR=0.05+0.1*rand(1,1);
                    CrPriods_Index(j)=1;
                
                else
                    CR=0.9+0.1*rand(1,1);
                    CrPriods_Index(j)=2;    
                end
                CrPriods_Count(CrPriods_Index(j))=CrPriods_Count(CrPriods_Index(j)) + 1;
            else
                 if (Ali<=NW(1))
                    CR=0.05+0.1*rand(1,1);
                    CrPriods_Index(j)=1;
                
                 else
                    CR=0.9+0.1*rand(1,1);
                    CrPriods_Index(j)=2;    
                end
                CrPriods_Count(CrPriods_Index(j))=CrPriods_Count(CrPriods_Index(j)) + 1;
            end

            %%%%%%%%%%%%%%%%%END OF CR RULE%%%%%%%%%%%%%%%%%%%%%%%%%%%
                f=Fit;
                [~, in]=sort(f,'ascend');
                AA=[in(1) in(2) in(3)  in(4) in(5)  ];
                BB=[in(46) in(47) in(48) in(49) in(50) ];
                CC=[in(6) in(7) in(8) in(9) in(10) in(11) in(12) in(13) in(14) in(15) in(16) in(17) in(18) in(19) in(20) in(21) in(22) in(23) in(24) in(25) in(26) in(27) in(28) in(29) in(30) in(31) in(32) in(33) in(34) in(35) in(36) in(37) in(38) in(39) in(40) in(41) in(42) in(43) in(44) in(45)];
               
            % choose three random individuals from population,
            % mutually different 

           paraIndex=floor (rand(1,1)*length(AA))+1;
                paraIndex1=floor (rand(1,1)*length(BB))+1;
                paraIndex2=floor (rand(1,1)*length(CC))+1;
                r(1) = AA(paraIndex);
                r(2) = BB(paraIndex1);
                r(3) = CC(paraIndex2);

                F=0.1+0.9*rand(1,1);
               
                Rnd = floor(rand()*D) + 1;
                for i = 1:D
                    if ( rand()<CR ) || ( Rnd==i )
                        X(i)=Pop(i,r(3))+F*(Pop(i,r(1))-(Pop(i,r(2))));
                    else
                        X(i) = Pop(i,j);
                    end
                end
        % end%end of All cases
        
        % verify boundary constraints
        % verify boundary constraints
        for i = 1:D
            if (X(i) < L ) || ( X(i) > H )
                X(i) = L + (H-L)*rand();
            end
        end
        
        % select the best individual
        % between trial and current ones
        % calculate fitness of trial individual   
         f=testFunction(X, train, test, k); 
        % if trial is better or equal than current
        if f <= Fit(j)
            % CRRatio(find(A==CRs(j)))=CRRatio(find(A==CRs(j)))+1-(min(f,Fit(j))/max(f,Fit(j)));
           Sr (CrPriods_Index(j)) = Sr(CrPriods_Index(j)) +1;
            Pop(:,j) = X; % replace current by trial
            Fit(j) = f ;
            % if trial is better than the best
            if f <= Fit(iBest)
                iBest = j ; % update the best�s index
            end
        else
        end
    end
    CrPriods_Count(CrPriods_Count==0)=0.0001;
    Sr=Sr./CrPriods_Count;
%%%%%%%%%%%%%%%%USING SR ONLY%%%%%%%%%%5    
    if(sum(Sr)==0)
        W = [1/2 1/2];
    else
        W = Sr/sum(Sr);
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%5
    NW=(NW*(g-1)+W)/g;
    Cr_All=Cr_All+CrPriods_Count;
end

bestSolution=Pop(:,iBest);
bestFitness = Fit(iBest);
iter = g;

end