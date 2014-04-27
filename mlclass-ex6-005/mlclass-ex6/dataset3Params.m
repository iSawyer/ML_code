function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.3;
sigma = 0.1;
temp_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
temp_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, ];
m = size(temp_C,2);
n = size(temp_sigma,2);
%model = zeros(size(temp_C,2),size(temp_sigma,2));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
x1 = [1 2 1]; x2 = [0 4 -1];
for i = 1 : m
    for j = 1: n
        model(i,j) = svmTrain(X, y,temp_C(i), @(x1, x2) gaussianKernel(x1, x2, temp_sigma(j)));
    end
end
err = zeros(m,n);
for i = 1:m
    for j=1:n
        predictions = svmPredict(model(i,j), Xval);
        err(i,j) = mean(double(predictions ~= yval));
    end
end

[id1,id2] = find(err == min(min(err))); 



C= temp_C(id1);
sigma = temp_sigma(id2);







% =========================================================================

end
