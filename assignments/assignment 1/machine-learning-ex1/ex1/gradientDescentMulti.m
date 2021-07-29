function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % ITERATIVE
    feat = size(X, 2);
    temp_sum = zeros(feat, 1);
    
    for i = 1:m,
        for j = 1:feat,
            temp_sum(j) = temp_sum(j) + (theta' * (X(i, :)') - y(i)) * X(i, j);   
        end
    end
    
    theta = theta - ((alpha / m) * temp_sum);

    % VECTORIZATION
##    theta = theta - (alpha / m) * (X' * (X * theta - y));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
