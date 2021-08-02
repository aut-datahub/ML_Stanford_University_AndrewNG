function cost = testSetCost(X, y, Xtest, ytest, lambda)
  
  p = 8;
  m = size(X, 1);
  mtest = size(Xtest, 1);
  
  X_poly = polyFeatures(X, p);
  [X_poly, mu, sigma] = featureNormalize(X_poly);
  X_poly = [ones(m, 1), X_poly];
  
  X_poly_test = polyFeatures(Xtest, p);
  X_poly_test = bsxfun(@minus, X_poly_test, mu);
  X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
  X_poly_test = [ones(mtest, 1), X_poly_test];
  
  [theta] = trainLinearReg(X_poly, y, lambda);
  
  [cost, grad] = linearRegCostFunction(X_poly_test, ytest, theta, 0);
  
end