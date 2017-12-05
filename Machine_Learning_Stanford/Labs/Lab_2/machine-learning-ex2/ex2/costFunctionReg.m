function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

lambda = 0.1

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Sum terms
JSumTerm = 0;
regSumTerm = 0;
gradSumTerm = zeros(size(theta));

% Calculate hypothesis
hyp = sigmoid(transpose(theta) * transpose(X));

% Calculate the gradient sum terms
for i = 1:m,
  JSumTerm = JSumTerm + ((-y(i) * log(hyp(i))) - ((1 - y(i)) * (log(1 - hyp(i)))));

  gradSumTerm(1) = gradSumTerm(1) + (hyp(i) - y(i)) * X(i, 1);

  for j = 2:size(theta),
    gradSumTerm(j) = gradSumTerm(j) + (hyp(i) - y(i)) * X(i, j);
  end
end

% Calculate the final gradient terms
grad(1) = (1/m) * gradSumTerm(1);
for j = 2:size(theta),
  grad(j) = ((1/m) * gradSumTerm(j)) + (lambda/m)*(theta(j));
  regSumTerm = regSumTerm + (theta(j))^2;
end

% Calculate the error
J = (1/m) * (JSumTerm) + (lambda/(2*m))*regSumTerm;





% =============================================================

end
