function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = zeros(64, 3);

for c = 1:length(values),
  for sig = 1:length(values),
    model= svmTrain(X, y, values(c), @(x1, x2) gaussianKernel(x1, x2, values(sig)));
    predictions = svmPredict(model, Xval);
    results(8*(c-1) + sig, :) = [values(c), values(sig), mean(double(predictions ~= yval))]; 
  endfor;
endfor;

[value index] = min(results(:, 3));

C = results(index, 1);
sigma = results(index, 2);

fprintf("C = %f, sigma = %f, Minimum Cost = %f", C, sigma, value);

% =========================================================================

end
