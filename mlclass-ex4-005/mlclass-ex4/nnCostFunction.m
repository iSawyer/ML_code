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
%

% 给X中添加列
temp_1h = zeros(1,10);
temp_2h = zeros(1,10);
weight_1 = 0.0;
weight_2 = 0.0;
temp_Theta1 = Theta1(:,2:end);
temp_Theta2 = Theta2(:,2:end);
tran_1 = zeros(size(Theta1));
tran_2 = zeros(size(Theta2));
X = [ones(m,1),X];  % X ~ 5000,401
z_1 = X * Theta1';
z_1 = [ones(m,1),z_1];
h_x_1 = sigmoid(X * Theta1'); % h_x_1 ~ 5000,25
h_x_1 = [ones(m,1),h_x_1]; % 5000,26
h_x = sigmoid (h_x_1 * Theta2'); % 5000,10
for j = 1 : size(temp_Theta1,1)
 for k = 1 : size(temp_Theta1,2)
  weight_1 = weight_1 + temp_Theta1(j,k) * temp_Theta1(j,k);
 end
end
for j = 1 : size(temp_Theta2,1)
 for k = 1 : size(temp_Theta2,2)
  weight_2 = weight_2 + temp_Theta2(j,k) * temp_Theta2(j,k);
 end
end 
for i = 1 : size(X,1)
  y_temp = 1 : num_labels;
  y_temp = (y_temp == y(i));
  temp_1h = log(h_x(i,:));
  temp_2h = log(1 - h_x(i,:));
  J = J - sum( y_temp .* temp_1h + (1-y_temp) .* temp_2h );
  delta_3 = ( h_x(i,:) - y_temp );  % 1 * 10
  delta_2 = (Theta2' * delta_3') .* sigmoidGradient(z_1(i,:)');  % 26 * 10 * 10 * 1  ~ 26 * 1
  tran_1 = tran_1 + delta_2( 2:end ) * X(i,:);    % 25*1 * 1*401 = 25 * 401
  tran_2 = tran_2 + delta_3' * h_x_1(i,:);  % 10*1 * 1 * 26 = 10 * 26
end
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));
% Theta1 ~ 25 * 401
% Theta2 ~ 10 * 26
J = J / m ;
J = J + ( weight_1 + weight_2 ) * lambda / (2 * m );
Theta1_temp = [zeros(size(Theta1,1),1) , Theta1(:,2:end)];
Theta2_temp = [zeros(size(Theta2,1),1), Theta2(:,2:end)];
Theta1_grad = tran_1 / m + Theta1_temp * lambda / m;
Theta2_grad = tran_2 / m + Theta2_temp * lambda / m;




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
