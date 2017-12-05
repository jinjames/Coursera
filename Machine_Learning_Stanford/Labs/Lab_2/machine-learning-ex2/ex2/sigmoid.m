function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


% Input is a scalar value
if numel(z) == 1,
  g = 1/(1+(e^(-z)))

% Input is a vector or matrix
else,
  for index = 1:numel(z),
    g(index) = 1/(1+(e^(-z(index))))
end

% =============================================================

end
