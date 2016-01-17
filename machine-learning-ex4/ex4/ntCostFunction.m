function [J grad] = ntCostFunction(nn_params, ...
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

            % L1 -> L2 (a2 z2) -> L3 (a3 z3)
            % size(X)
            % size(Theta1)
            % size(Theta2)
            % Add ones to the X data matrix
            X_ = [ones(m, 1) X];
            % size(X_)
            % don't forget sigmoid function
            z2 = X_ * Theta1';
            % size(a2)
            a2 = sigmoid(z2);
            % size(z2)
            a2_ = [ones(m, 1) a2];
            % z2_ = [ones(size(Theta1',1), 1) z2];
            % size(z2_)
            z3 = a2_ * Theta2';
            % size(a3)
            a3 = sigmoid(z3);
            % size(z3)
            % size(y)
            decodey = [];
            for i = 1:m
                temp = zeros(num_labels,1);
                % size(temp)
                temp(y(i)) = 1;
                decodey = [decodey temp];
            end
            % decodey
            % size(decodey)
            tempJ = (decodey' .* log (a3)) + ((1-decodey)' .* log(1-a3));
            J = - (sum(sum(tempJ))) / m;

            % regularization
            % Notice that you can first compute the unregularized cost function J using your existing nnCostFunction.m and then later add the cost for the regularization terms.
            % Note that you should not be regularizing the terms that correspond to the bias.
            % size(Theta1)
            % size(Theta2)
            % difference of (:,2:end), (2:end) on vector / matrix
            % sum is automatically applied for matrix
            regularization = (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)))*lambda/m/2;
            J =  J + regularization;
            % guess loop is easier to implement
            % vectorization is fun and challenging to implement
            % cost = 0;
            % for i = 1:m
            %     for k = 1:num_labels

            %     end
            % end

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
            % this shoud be a hint for Part 1 ...
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
            % size(a3)
            % size(decodey')
            delta_3 = a3 - decodey';
            % size(Theta2)
            % size(a2_)
            temp = (delta_3 * Theta2);
            delta_2 = temp(:,2:end) .* sigmoidGradient(z2);
            % 5000 samples
            % size(delta_2)
            % size(delta_3)
            % size(a2')
            % size(a3')
            D2 = a2' * delta_3;
            D1 = X' * delta_2;
            % D1 = sum(delta_2 * X',2)
            % D2 = sum(delta_3 * a2',2)
            % D1 = 
            % for i = 1:m
            %     for k = 1:num_labels

            %     end
            % end
            % size(D2)
            % size(D1)

            % ~~ debug ~~
            % embarrasing... 
                % the matrix order 25*10 or 10*25 (and now it's correct) D1 / D1'
                % because when it's unrolled, you can't tell
            % the bias entry... why not 1?
            % the m is divided into all terms
            % the origin of everthing
            % file:///media/EOS_DIGITAL/Machine%20Learning/screenshots/9%20-%202%20-%20Backpropagation%20Algorithm%20(12%20min).mp4-2013-11-25-09h02m54s157.png
            Theta1_grad = D1';
            Theta2_grad = D2';

            % add the ones column for checkNNGradients
            % size(delta_2)
            % size(delta_3)
            bias1 = sum(delta_2, 1);
            bias2 = sum(delta_3, 1);
            Theta1_grad = [bias1' Theta1_grad] / m;
            Theta2_grad = [bias2' Theta2_grad] / m;
            % Theta1_grad = [ones(size(Theta1_grad,1), 1) Theta1_grad];
            % Theta2_grad = [ones(size(Theta2_grad,1), 1) Theta2_grad];


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
%             % add row
%                 % [zeros(1,3); magic(3)(2:end,:)]
%             % add column
%                 % [zeros(3,1) magic(3)(:,2:end)]
%             % size(Theta1)
%             % size(Theta1_grad)
%             % size(Theta2)
%             % size(Theta2_grad)
%             temp_theta1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
%             temp_theta2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
%             % size(Theta1_grad)
%             % size(temp_theta1)
%             % size(Theta2_grad)
%             % size(temp_theta2)
%             temp_theta1 = temp_theta1*lambda/m;
%             temp_theta2 = temp_theta2*lambda/m;
%             Theta1_grad += temp_theta1;
%             Theta2_grad += temp_theta2;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end