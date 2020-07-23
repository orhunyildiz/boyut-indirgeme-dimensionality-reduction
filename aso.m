function [bestSolution, bestFitness, iteration]=aso(train, test, maxIteration, knn)

settings;

Atom_Num=50;
Max_Iteration=ceil(maxIteration/Atom_Num)+1;
Low = 0;
Up = 1;
[~, dimension] = size(train);
Dim = dimension - 1;
alpha=50;
beta=0.2;
   % Randomly initialize positions and velocities of atoms.
     if size(Up,2)==1
         Atom_Pop=rand(Atom_Num,Dim).*(Up-Low)+Low;
         Atom_V=rand(Atom_Num,Dim).*(Up-Low)+Low;
     end
   
     if size(Up,2)>1
        for i=1:Dim
           Atom_Pop(:,i)=rand(Atom_Num,1).*(Up(i)-Low(i))+Low(i);
           Atom_V(:,i)=rand(Atom_Num,1).*(Up(i)-Low(i))+Low(i);
        end
     end

     Fitness=zeros(1,Atom_Num);
 % Compute function fitness of atoms.
     for i=1:Atom_Num
       Fitness(i)=testFunction(Atom_Pop(i,:)', train, test, knn);
     end
       Functon_Best=zeros(Max_Iteration,1);
       [~,Index]=min(Fitness);
       Functon_Best(1)=Fitness(Index);
       X_Best=Atom_Pop(Index,:);
     
 % Calculate acceleration.
 Iteration=1;
 Atom_Acc=Acceleration(Atom_Pop,Fitness,Iteration,Max_Iteration,Dim,Atom_Num,X_Best,alpha,beta);

 % Iteration
 for Iteration=2:Max_Iteration 
           Functon_Best(Iteration)=Functon_Best(Iteration-1);
           Atom_V=rand(Atom_Num,Dim).*Atom_V+Atom_Acc;
           Atom_Pop=Atom_Pop+Atom_V;     
    
         for i=1:Atom_Num
       % Relocate atom out of range.  
           TU= Atom_Pop(i,:)>Up;
           TL= Atom_Pop(i,:)<Low;
           Atom_Pop(i,:)=(Atom_Pop(i,:).*(~(TU+TL)))+((rand(1,Dim).*(Up-Low)+Low).*(TU+TL));
           %evaluate atom. 
           Fitness(i)=testFunction(Atom_Pop(i,:)', train, test, knn);
         end
        [Max_Fitness,Index]=min(Fitness);      
     
        if Max_Fitness<Functon_Best(Iteration)
             Functon_Best(Iteration)=Max_Fitness;
             X_Best=Atom_Pop(Index,:);
          else
            r=fix(rand*Atom_Num)+1;
             Atom_Pop(r,:)=X_Best;
        end
     
      % Calculate acceleration.
       Atom_Acc=Acceleration(Atom_Pop,Fitness,Iteration,Max_Iteration,Dim,Atom_Num,X_Best,alpha,beta);
 end

bestSolution=X_Best;
bestFitness=Functon_Best(Iteration);
iteration=(Iteration-1)*Atom_Num;
end

function Acc=Acceleration(Atom_Pop,Fitness,Iteration,Max_Iteration,Dim,Atom_Num,X_Best,alpha,beta)
%Calculate mass 
  M=exp(-(Fitness-max(Fitness))./(max(Fitness)-min(Fitness)));
  M=M./sum(M);  
  
 
    G=exp(-20*Iteration/Max_Iteration); 
    Kbest=Atom_Num-(Atom_Num-2)*(Iteration/Max_Iteration)^0.5;
    Kbest=floor(Kbest)+1;
    [~, Index_M]=sort(M,'descend');
 
 E=zeros(Atom_Num,Dim);
 a=zeros(Atom_Num,Dim);
 for i=1:Atom_Num       
     E(i,:)=zeros(1,Dim);   
   MK(1,:)=sum(Atom_Pop(Index_M(1:Kbest),:),1)/Kbest;
   Distance=norm(Atom_Pop(i,:)-MK(1,:),2);   
     for ii=1:Kbest
                  j=Index_M(ii);       
                   %Calculate LJ-potential
                  Potential=LJPotential(Atom_Pop(i,:),Atom_Pop(j,:),Iteration,Max_Iteration,Distance);                   
                  E(i,:)=E(i,:)+rand(1,Dim)*Potential.*((Atom_Pop(j,:)-Atom_Pop(i,:))/(norm(Atom_Pop(i,:)-Atom_Pop(j,:))+eps));             
     end

        E(i,:)=alpha*E(i,:)+beta*(X_Best-Atom_Pop(i,:));
        %calculate acceleration
     a(i,:)=E(i,:)./M(i); 
 end
Acc=a.*G;
end

function Potential=LJPotential(Atom1,Atom2,Iteration,Max_Iteration,s)
 %Calculate LJ-potential
r=norm(Atom1-Atom2,2);  
c=(1-(Iteration-1)/Max_Iteration).^3;  
%g0=1.1;
%u=2.4;
rsmin=1.1+0.1*sin(Iteration/Max_Iteration*pi/2);
rsmax=1.24;

if r/s<rsmin
    rs=rsmin;
else
    if  r/s>rsmax
        rs=rsmax;  
    else
        rs=r/s;
    end
end           
 
Potential=c*(12*(-rs)^(-13)-6*(-rs)^(-7)); 
end



