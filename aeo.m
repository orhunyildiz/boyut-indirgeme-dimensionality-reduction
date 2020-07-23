function [bestSolution, bestFitness, iter]=aeo(train, fNumber, maxIteration, knn)
settings;
Low = 0;
Up = 1;
[~, dimension] = size(train);
Dim = dimension - 1;
nPop = 50;
MaxIt = ceil((maxIteration/ (nPop*2)));
newPopFit = zeros(1,nPop);
PopFit = zeros(1,nPop);
PopPos = zeros(nPop, Dim);

for i=1:nPop   
        PopPos(i,:)=rand(1,Dim).*(Up - Low) + Low;
        PopFit(i)=testFunction(PopPos(i,:)', train, fNumber, knn); 
end
[~, indF]=sort(PopFit,'descend');

PopPos=PopPos(indF,:);
PopFit=PopFit(indF);

BestF=PopFit(end);
BestX=PopPos(end,:);

Matr=[1,Dim];

for It=1:MaxIt    
    r1=rand;
    a=(1-It/MaxIt)*r1;
    xrand=rand(1,Dim).*(Up-Low)+Low;
    newPopPos(1,:)=(1-a)*PopPos(nPop,:)+a*xrand; %equation (1)
         
    u=randn(1,Dim);
    v=randn(1,Dim);  
    C=1/2*u./abs(v); %equation (4)
    newPopPos(2,:)=PopPos(2,:)+C.*(PopPos(2,:)-newPopPos(1,:)); %equation (6)
 
for i=3:nPop

    u=randn(1,Dim);
    v=randn(1,Dim);  
    C=1/2*u./abs(v);  
    r=rand;
    if r<1/3
        newPopPos(i,:)=PopPos(i,:)+C.*(PopPos(i,:)-newPopPos(1,:)); %equation (6)
    else
        if 1/3<r<2/3            
            newPopPos(i,:)=PopPos(i,:)+C.*(PopPos(i,:)- PopPos(randi([2 i-1]),:)); %equation (7)

        else    
            r2=rand;  
            newPopPos(i,:)=PopPos(i,:)+C.*(r2.*(PopPos(i,:)- newPopPos(1,:))+(1-r2).*(PopPos(i,:)-PopPos(randi([2 i-1]),:))); %equation (8)
        end
    end
end       
        
         for i=1:nPop        
             newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
             newPopFit(i)=testFunction(newPopPos(i,:)', train, fNumber, knn);  
                if newPopFit(i)<PopFit(i)
                   PopFit(i)=newPopFit(i);
                   PopPos(i,:)=newPopPos(i,:);
                end
         end
         
         [~, indOne]=min(PopFit);
     for i=1:nPop
            r3=rand;   Ind=round(rand)+1;
    newPopPos(i,:)=PopPos(indOne,:)+3*randn(1,Matr(Ind)).*((r3*randi([1 2])-1)*PopPos(indOne,:)-(2*r3-1)*PopPos(i,:)); %equation (9)
      end
     
      for i=1:nPop        
             newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
             newPopFit(i)=testFunction(newPopPos(i,:)', train, fNumber, knn);     
                if newPopFit(i)<PopFit(i)
                   PopPos(i,:)=newPopPos(i,:);
                    PopFit(i)=newPopFit(i);
                end
      end
         
      [~,indF]=sort(PopFit,'descend');

      PopPos=PopPos(indF,:);
      PopFit=PopFit(indF);
        if PopFit(end)<BestF
            BestF=PopFit(end);
            BestX=PopPos(end,:);            
        end
end 
bestSolution = BestX;
bestFitness = BestF;
iter = It*(nPop*2);
end

function  X=SpaceBound(X,Up,Low)
    Dim=length(X);
    S=(X>Up)+(X<Low);    
    X=(rand(1,Dim).*(Up-Low)+Low).*S+X.*(~S);
end




