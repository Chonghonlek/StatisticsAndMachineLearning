Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

Linear regression is useful for predicting quantitative response. 

To assess the accuracy of the model we can use 


Suppose we want to answer these questions:

1) Is There a Relationship Between the Response and Predictors?
- we apply the f-test aka one way anova.
- We apply partial test to study the partial effect of adding that variable to the model
- We still look at overall F-statistic instead of individual p-values. Consider p=100 and H0 is true, then about 5% of the p-values of each variable will be below 0.05 by chances. We expect to see 5 small p-values even in the absence of true association between predictor and response. Thus we still look at F-statistic - as it adjusts for the number of predictors. 
  
2) deciding on important variables 
 - Variable selection will be studied separately. 
 - Some statistics can be used to judge quality of model like : Mallow’s Cp, Akaike information criterion (AIC), Bayesian information criterion (BIC), and adjusted R^2
 - There are a total of $2^k$ that contain subsets of k variables. Unless k is very small, we cant consider all models. 3 classical approaches to this is : Forward Selection, Backward Selection and Mixed selection

3) Model fit
- the residual standard error (RSE or MSres)
- R^2 and adjuisted R^2 statisitic - measure of fit. it is also a measure of linear relationship between X and Y
- R^2 close to 1 means the model explains a large portion of the variance in the response variable. 

4) Predictions
- the least squares plane : $\hat{Y} = \hat{\beta_0} + \cdots + \hat{\beta_k}X_k$ is only an estimate for the true population regression plane f(X). The inaccurary in the coefficient estimates is related to reducible error. We can compute a CI to determine how close $\hat{Y}$ is to f(X)
- A source of potentially reducible error is model bias. this happens when we use a linear model to do an estimate that lands us the best linear approximation to true surface
- Even if we knew the true value for $\beta$ we may not predict perfectly due to the random error induced by $\epsilon$ - this is the irreducible error. To answer how much Y will vary from $\hat{Y}$, we use Prediction Intervals. 
- PI is larger than CI given that PI incorporate both the error in the estimat for f(X) and the uncertainty to how much an individual point differs from true population regression plane.