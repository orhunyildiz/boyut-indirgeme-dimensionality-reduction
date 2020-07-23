function [bestSolution, bestFitness, iteration]=boa(train, test, maxIteration, knn)

settings;

n = 50;
N_iter = ceil(maxIteration/(n*2));
Lb = 0;
Ub = 1;

[~, dimension] = size(train);
dim = dimension - 1;
p = 0.8;                       % probabibility switch
power_exponent = 0.1;
sensory_modality = 0.01;

%Initialize the positions of search agents
Sol = initialization(n,dim,Ub,Lb);
Fitness = zeros(1, n);
for i=1:n
    Fitness(i)=testFunction(Sol(i,:)', train, test, knn);
end

% Find the current best_pos
[fmin,I]=min(Fitness);
best_pos=Sol(I,:);
S=Sol; 

% Start the iterations -- Butterfly Optimization Algorithm 
for t=1:N_iter
  
        for i=1:n % Loop over all butterflies/solutions
         
          %Calculate fragrance of each butterfly which is correlated with objective function
          Fnew=testFunction(S(i,:)', train, test, knn);
          FP=(sensory_modality*(Fnew^power_exponent));   
    
          %Global or local search
          if rand<p   
            dis = rand * rand * best_pos - Sol(i,:);        %Eq. (2) in paper
            S(i,:)=Sol(i,:)+dis*FP;
           else
              % Find random butterflies in the neighbourhood
              epsilon=rand;
              JK=randperm(n);
              dis=epsilon*epsilon*Sol(JK(1),:)-Sol(JK(2),:);
              S(i,:)=Sol(i,:)+dis*FP;                         %Eq. (3) in paper
          end
           
            % Check if the simple limits/bounds are OK
            S(i,:)=simplebounds(S(i,:),Lb,Ub);
          
            % Evaluate new solutions
            Fnew=testFunction(S(i,:)', train, test, knn);  %Fnew represents new fitness values
            
            % If fitness improves (better solutions found), update then
            if (Fnew<=Fitness(i))
                Sol(i,:)=S(i,:);
                Fitness(i)=Fnew;
            end
           
           % Update the current global best_pos
           if Fnew<=fmin
                best_pos=S(i,:);
                fmin=Fnew;
           end
         end
                     
         %Update sensory_modality
          sensory_modality=sensory_modality_NEW(sensory_modality, N_iter);
end
bestSolution=best_pos;
bestFitness=fmin;
iteration=t*2*n;
end

% Boundary constraints
function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb;
  
  % Apply the upper bounds 
  J=ns_tmp>Ub;
  ns_tmp(J) = Ub;
  % Update this new move 
  s=ns_tmp;
end
  
function y=sensory_modality_NEW(x,Ngen)
    y=x+(0.025/(x*Ngen));
end

function [X]=initialization(N,dim,up,down)
    X = zeros(N, dim);
    for i=1:dim
        high = up;
        low = down;
        X(:,i)=rand(1,N).*(high-low)+low;
    end
end


