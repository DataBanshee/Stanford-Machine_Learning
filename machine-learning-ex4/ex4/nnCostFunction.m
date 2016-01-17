function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X,2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

m = size(X,1);
y_check = zeros(m,num_labels);
for i=1:m
    y_check(i,y(i))= 1;
end


X_biais = [ones(m,1) X];
a_1 = X_biais;

% Forwardpropagation

z_2 = Theta1*X_biais';
a_2 = sigmoid(z_2);
a_2_ = [ones(1,size(z_2,2));sigmoid(z_2)];

z_3 = Theta2*a_2_;
a_3 = sigmoid(z_3);

regularization_term = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2)))+(sum(sum(Theta2(:,2:end).^2))));% no term Theta 0



leftPart = -y_check.*log(a_3)';

rightPart = -(1-y_check).*log(1-a_3)';

J = (1/m)*sum(sum(leftPart+rightPart,2))+regularization_term;

%% Part 2 : Backpropagation
% 
% for t=1:m
%    
%     a1_ = [1 X(t,:)]; %1*401
%     z2 = Theta1*a1_';%25*1
%     a2 = sigmoid(z2);%25*1
%     a2_biais = [1;a2];%26*1
%     z3 = Theta2*a2_biais;%10*26..26*1
%     a3 = sigmoid(z3); %10*1
%     yy = ([1:num_labels]==y(t))';
%     
%     delta_3 = a3-y_check(t,:)';%10*1
%     delta_2 = (Theta2'*delta_3).*[1;sigmoidGradient(z2)];%26*1
%     delta_2 =delta_2(2:end,1); %25*1
%     
%     %Gradient Accumulation
%     
%     Theta1_grad = Theta1_grad + delta_2*a1_; %25*401
%     Theta2_grad = Theta2_grad + delta_3*a2_biais'; %10*26
%     
% end
% 
% Theta1_grad = Theta1_grad /m; %25*401
% Theta2_grad = Theta2_grad /m; %10*26

    delta_3 = a_3'-y_check; % Size 5000*10
    temp =(delta_3*Theta2)'; % Size 5000*26
    delta_2 = temp(2:end,:).*sigmoidGradient(z_2);% Size 5000*25
    delta_1 = Theta1'*delta_2;% Size 401*5000


% Gradient Accumulation 



Theta2_grad= a_2_*delta_3; %26*10 %25*10
Theta1_grad = (delta_2*X_biais)'; %401*25 %400*25

Theta2_grad= Theta2_grad'; %10*26
Theta1_grad = Theta1_grad'; %25*401


% Theta2_grad = (1/m).*[ones(size(Theta2_grad,1),1) Theta2_grad];
% Theta1_grad = (1/m).*[ones(size(Theta1_grad,1),1) Theta1_grad];    
% bias1 = sum(delta_2', 1);
%             bias2 = sum(delta_3, 1);
%             Theta1_grad = [bias1' Theta1_grad] / m;
%             Theta2_grad = [bias2' Theta2_grad] / m;
Theta1_grad = (Theta1_grad+lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]) / m;
Theta2_grad = (Theta2_grad+lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]) / m;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end