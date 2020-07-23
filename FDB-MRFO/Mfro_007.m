%--------------------------------------------------------------------------
% MRFO code v1.0.
% Developed in MATLAB R2011b
% The code is based on the following papers.
% W. Zhao, Z. Zhang and L. Wang, Manta ray foraging optimization: An
% effective bio-inspired optimizer for engineering applications.
% Engineering Applications of Artifical Intelligence (2019),
% https://dio.org/10.1016/j.engappai.2019.103300.
% -------------------------------------------------------------------------
% FunIndex: Index of function.
% MaxIt: The maximum number of iterations.
% PopSize: The size of population.
% PopPos: The position of population.
% PopFit: The fitness of population.
% Dim: The dimensionality of prloblem.
% Alpha: The weight coefficient in chain foraging.
% Beta: The weight coefficient in cyclone foraging.
% S: The somersault factor.
% BestF: The best fitness corresponding to BestX. 
% HisBestFit: History best fitness over iterations. 
% Low: The low bound of search space.
% Up: The up bound of search space.

function [bestSolution, bestFitness, iter]=Mfro_007(train, test, maxIteration, k)
settings;
[~, dimension] = size(train);
Low = 0;
Up = 1;
Dim = dimension - 1;
nPop = 50;
MaxIt = ceil((maxIteration/ (nPop*2)));

    for i=1:nPop   
        PopPos(i,:) = rand(1,Dim).*(Up-Low)+Low;
        PopFit(i) = testFunction(PopPos(i,:)', train, test, k);   
    end
       BestF=inf;
       BestX=[];

    for i=1:nPop
        if PopFit(i)<=BestF
           BestF=PopFit(i);
           BestX=PopPos(i,:);
        end
    end
    
fitnessOnce = BestF;
fitnessKontrol = 0;
for It=1:MaxIt  
     Coef=It/MaxIt; 
       if rand<0.5
          r1=rand;                         
          Beta=2*exp(r1*((MaxIt-It+1)/MaxIt))*(sin(2*pi*r1));    
          if  Coef>rand                                                      
              newPopPos(1,:)=BestX+rand(1,Dim).*(BestX-PopPos(1,:))+Beta*(BestX-PopPos(1,:)); %Equation (4)
          else
              IndivRand=rand(1,Dim).*(Up-Low)+Low;                                
              newPopPos(1,:)=IndivRand+rand(1,Dim).*(IndivRand-PopPos(1,:))+Beta*(IndivRand-PopPos(1,:)); %Equation (7)         
          end              
       else 
            Alpha=2*rand(1,Dim).*(-log(rand(1,Dim))).^0.5;           
            newPopPos(1,:)=PopPos(1,:)+rand(1,Dim).*(BestX-PopPos(1,:))+Alpha.*(BestX-PopPos(1,:)); %Equation (1)
       end
       fdbIndex = fitnessDistanceBalance( PopPos, PopFit);
    for i=2:nPop
        if rand<0.5
           r1=rand;                         
           Beta=2*exp(r1*((MaxIt-It+1)/MaxIt))*(sin(2*pi*r1));    
             if  Coef>rand 
                newPopPos(i,:)=BestX+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Beta*(BestX-PopPos(i,:)); %Equation (4)                               
             else
                 IndivRand=rand(1,Dim).*(Up-Low)+Low;
                    newPopPos(i,:)=IndivRand+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Beta*(IndivRand-PopPos(i,:));  %Equation (7)      
             end              
        else
            
            Alpha=2*rand(1,Dim).*(-log(rand(1,Dim))).^0.5;   
            if fitnessKontrol == 1
                newPopPos(i,:)=PopPos(i,:)+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Alpha.*(PopPos(fdbIndex,:)-PopPos(i,:)); %Equation (1)
            else
                newPopPos(i,:)=PopPos(i,:)+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Alpha.*(BestX-PopPos(i,:)); %Equation (1)
            end        
            
       end         
    end
         
           for i=1:nPop        
               newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
               newPopFit(i)=testFunction(newPopPos(i,:)', train, test, k);   
              if newPopFit(i)<PopFit(i)
                 PopFit(i)=newPopFit(i);
                 PopPos(i,:)=newPopPos(i,:);
              end
           end
           
           
            [bestFit, ~] = min(PopFit);

           fitnessKontrol = 0;
            if abs(fitnessOnce-bestFit)<1
                fitnessKontrol = 1;
            end
            fitnessOnce = bestFit;
           
            S=2;
        for i=1:nPop           
            newPopPos(i,:)=PopPos(i,:)+S*(rand*BestX-rand*PopPos(i,:)); %Equation (8)
        end
     
     for i=1:nPop        
         newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
         newPopFit(i) = testFunction(newPopPos(i,:)', train, test, k);     
         if newPopFit(i)<PopFit(i)
            PopFit(i)=newPopFit(i);
            PopPos(i,:)=newPopPos(i,:);
         end
     end
     
     for i=1:nPop
        if PopFit(i)<BestF
           BestF=PopFit(i);
           BestX=PopPos(i,:);            
        end
     end
     
        if abs(fitnessOnce-BestF)<1
            fitnessKontrol = 1;
        end
        fitnessOnce = BestF;
     
end

bestSolution = BestX;
bestFitness = BestF;
iter = It*(nPop*2);
