Notes taken from **Douglas C M et al, Introduction to Linear Regression Analysis**

We aim to 
- include as many regressor variables so that the information constraint in these variable can influence the predicted values
- include as few to get a simple model and reduce variance of y due to more regressors.

The process of finding a model that is a compormise between these 2 objectives is the variable selection problem

Full vs Subset Models
1) Full model can be written as $y = X_p\beta_p + X_r\beta_r + \epsilon$ where r is number of regressors variables to be deleted (p = K+1-r is retained), $X_p$ is the p variables to be retained. Subset model is given as: $y = X_p\beta_p +  \epsilon$
2) Full model least squares is $\hat{\beta} = (X^TX)X^Ty$ and the estimate of the residual variance is 

$$
\hat{\sigma^2} = \frac{y^Ty - \hat{\beta}^TX^Ty}{n-K-1} = \frac{y^T(I - X(X^TX)^{-1}X^T)y}{n-K-1}
$$

3) subset model: $\hat{\beta_p} = (X_p^TX_p)X_p^Ty$

4) The expected value of $\hat{\beta_p}$ is biased estimator unless $\beta_r$ or $X_p^T X_p =0$ (proof not shown)

$$
E(\hat{\beta_p}) = \beta_p + (X_p^T X_p)^{-1}X_p^T X_r\beta_r = \beta_p + A\beta_r \ne \beta_p
$$

5) $Var(\hat{\beta_p}) = \sigma^2 (X_p^TX_p)^{-1}$ and $Var(\hat{\beta}) = \sigma^2 (X^TX)^{-1}$
   - $Var(\hat{\beta}) - Var(\hat{\beta_p})$ is positive semi definite (proof not shown)
   - The variance in full model is greater than or eqaul to that of the variance in subset model.

6) $MSE(\hat{\beta_p}) = \sigma^2 (X_p^TX_p)^{-1} + A\beta_r\beta_r^TA^T$ (proof not shown)
- the matrix of $Var(\hat{\beta_p}) - MSE(\hat{\beta_p})$ is positive semidefinite if the matrix $Var(\hat{\beta_p}) -\beta_r\beta_r^T$ is positive semi definite.
- The least squares estimates in subset model have smaller mean square error than the corresponding parameter estimateor in full model. - only when deleted variables have regression coefficients smaller than standard error of their estimates in full model.

7) Estimate $\hat{\sigma^2} $ from full model is unbiased. but for subset model, 

$$
E(\hat{\sigma^2_p} ) = \sigma^2 + \frac{\beta_r^TX_r^T(I-X_p(X_p^TX_p)^{-1}X_p^T)X_r\beta_r}{n-p} \ne \sigma^2
$$

8) read book for comparisons for prediction ability

## Criteria for evaluating Subset Regression models
- $R^2$ or coefficient of multuple determination
- Adjusted $R^2 = 1- \frac{SS_{res}/(n-p)}{SS_t/(n-1)}$ 
- Residual mean square ($MS_{res} = \frac{SS_{res}}{n-p}$)
- Mallows's $C_p$ statistic
- AIC
  - $AIC = -2\ln(L) + 2p$ where p is number of parameters
  - in OLS Case:
$$
AIC = n\ln(\frac{SS_{res}}{n}) + 2p
$$

- $BIC = -2\ln(L) + p\ln(n)$
  - greater penalty on adding regressors as sample size increases

- others like press stastistic (depend on what you use the model for)

## All possible regression

1) fit all regression model and calculate criterion (like those above for each model)
2) Choose the "best" regression model using the criterion keeping in mind that simpler model is preferred

Note that if there K regressor variables, then there will be a total of $2^K$ fitted models to look through
- can be alot if there too many variables.
- in r

```r
library(olsrr)
fm = lm(y~x1 + x2+ x3, data = data)
all = ols_step_all_possible(fm)
plot(all)
```
## Forward selection

1) Start with no regressor variables in the model except the intercept. Choose a small $\alpha_{in}$ for determining whether a variable can be entered into model
2) Models with one variable are fitted. The p-value for testing the signifiance of the variable using $F = SS_R(x_j)/MS_{res}(x_j)$ for each model is calculated. Only variables that produce p-values smaller than $\alpha_{in}$  will be considered. We then choose the variable that produces the smallest p-value into the model
3) Suppose first variable is $x_1$. we then proceed with 2 variables to fit inclusive of $x_1$ The p-value for testing the significance of $x_j$ is calculated $F = SS_R(x_j|x_1)/MS_{res}(x_1,x_j)$
   - only variables smaller than $\alpha_{in}$ will be considered. 
   - the variable that produces smallest p-value is then entered as 2nd variable
4) Repeat until no more can be entered or last variable is entered.

```r
library(OLSRR)
OLS_step_forward_p(fm,penter = 0.05)
```

## backward selection

1) Start with all the K regressor variables including the intercept. Choose a small $\alpha_{out}$ to determine which varibale to remove
2) The p-vallue for testing the significance of each variable $x_j$ as if it is the last variable to enter model is calculated. Only variables that produce p_values larger than $\alpha_{out}$ are cosndiered for removal. we then remove the largest p-value variable. (note that f-statistic here is different from forward selection)
3) The procedure is then repeated with a model of K-1. 
4) This terminates when the largest p-valueis less than or equal to $\alpha_{out}$

```r
library(OLSRR)
OLS_step_backward_p(fm,prem = 0.05)
```

## Step wise regression

This combines both procedures

1) The procedure starts with no regressor variables. We choose a small $\alpha_{in}$ and a $\alpha_{out}$ to decide what variables to enter and remove.
2) We perform forward selection to choose the first regressor variable to be entered into the model. Assume that it is $x_1$
3) Perfrom forward selection for the 2nd regressor to enter. assume that it is $x_4$
4) Perfrom backward selection to determine if $x_1$ can be removed. It wil be removed if it becomes redudant due to its relationship with $x_4$
5) The procedure then continues with a forward selection. When new variable is entered, all the other variables in model is checked with backward selection to see if any of them can be removed
6) procedure terminates when no more variable can removed or entered

```r
library(OLSRR)
OLS_step_both_p(fm,pent = 0.15,prem = 0.15)
```