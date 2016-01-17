% clc;
close all;

nb_iter = 150;
theta = zeros(size(X,2),1); 
alpha = 0.03;

for i=1:nb_iter
    
   [cost, grad] = costFunction(theta, X, y);
   theta = theta - (alpha/m).*grad;
   
   J_Cout(i) = cost;
  
    
end

figure,plot(J_Cout);

fprintf('Optimal value of Thera : %f \n', theta);

%%
%%%%%%%%% Normal Equation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta_normal = inv(X'*X)*X'*y;