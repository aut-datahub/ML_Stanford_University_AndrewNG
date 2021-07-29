function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = g + (1 ./ (1 + exp(-z)));

##for i= 1:size(z)(1)
##  g(i) = g(i)  + ( 1 / (1 + exp(-z(i))));
##endfor


% =============================================================

end
