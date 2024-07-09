Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

# Logistic Regression

To avoid the problem of probability falling beyond $[0,1]$, we model $p(X)$ using a function that gives the outputs between 0 and 1 for all values of X. 

We use the logistic function:

$$
p(X) = \frac{e^{\beta_0 + \beta_1X}}{1 +e^{\beta_0 + \beta_1X} } 
$$

The logistic fucntion will always produce an "S" like curve. Also,

$$
\begin{aligned}
p(X) = \frac{e^{\beta_0 + \beta_1X}}{1 +e^{\beta_0 + \beta_1X} } &\iff \frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1X} \\
&\iff \log(\frac{p(X)}{1 - p(X)}) = \beta_0 + \beta_1X
\end{aligned}
$$

$\frac{p(X)}{1 - p(X)}$ is refeered as the odds and takes value between $(0,\infty)$, odds close to 0 imply low prob, vice versa.

$\log(\frac{p(X)}{1 - p(X)})$ is referred to as the log odds or _logit_ , logistic regression model has a logit that is linear.


To fit the model, we estimate the coefficients using the method of maximum likelihood. we define the likelihood function as 

$$
L(\beta_0,\beta_1) = \prod_{i : y_i = 1} p(x_i) \prod_{i': y_{i'} = 0} (1-p(x_{i'}))
$$

<details>

<summary>solve</summary>

```math
\begin{aligned}
l((\beta_0,\beta_1)) &= \sum \log p(x_i)+ \sum \log (1-p(x_{i'})) \\
&= \sum \log(e^{\beta_0 + \beta_1x_i}) - \log(1 + e^{\beta_0 + \beta_1x_i}) \\
&+ \sum \log(1) - \log(1 + e^{\beta_0 + \beta_1x_{i'}}) \\
&= \sum_{i : y_i = 1} (\beta_0 + \beta_1x_i) - \sum_i \log(1 + e^{\beta_0 + \beta_1x_i}) \\
\\
\frac{\partial l}{\partial \beta_0} &= \sum_{i : y_i = 1} 1 - \sum_i \frac{e^{\beta_0 + \beta_1x_i}}{1 + e^{\beta_0 + \beta_1x_i}} \rightarrow \text{set to } 0 \\
\\
\frac{\partial l}{\partial \beta_1} &= \sum_{i : y_i = 1} x_i - \sum_i \frac{x_ie^{\beta_0 + \beta_1x_i}}{1 + e^{\beta_0 + \beta_1x_i}} \rightarrow \text{set to } 0 \\
\end{aligned}
```

</details> <br>

The MLE for the parameters $\beta_0, \beta_1$ must fulfil the following equations. There is no closed form solution for the estimator. The solution can be computed using various algorithms like the newton raphson algorithm. (help from https://www.analyticsvidhya.com/blog/2022/02/decoding-logistic-regression-using-mle/)

$$
\begin{align}
\sum_i y_i &= \sum_i \frac{e^{\beta_0 + \beta_1x_i}}{1 + e^{\beta_0 + \beta_1x_i}} \\
\sum_i x_iy_i &=\sum_i \frac{x_ie^{\beta_0 + \beta_1x_i}}{1 + e^{\beta_0 + \beta_1x_i}}
\end{align}
$$

Once the coeffcients are have been estimated, we can compute the probability $p(X)$

For predictions, We can set a threshold eg p =0.5 s.t. if $p(x) > p = 0.5$ then $y_i = 1$ We may choose a lower threesholld for more conservative cases. 

## Multiple Logistic regression

here we generalise such that :

$$
\begin{aligned}
p(X) = \frac{e^{\beta_0 + \beta_1X_1 + \cdots \beta_pX_p}}{1 +e^{\beta_0 + \beta_1X+ \cdots \beta_pX_p} } &\iff \frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1X + \cdots \beta_pX_p} \\
&\iff \log(\frac{p(X)}{1 - p(X)}) = \beta_0 + \beta_1X + \cdots \beta_pX_p
\end{aligned}
$$

and we use MLE to estimate $\beta_0 ,\beta_1  \cdots \beta_p$

## Multinomial Logistic regression

In cases for more than 2 response classes i.e. K >2

We select a single class to serve as a baseline; WLOG, we select the Kth class for this role,

$$
\begin{aligned}
P(Y=k|X=x) &= \frac{e^{\beta_{Kk0} + \beta_{k1}X_1 + \cdots \beta_{kp}X_p}}{1 + \sum_{l=1}^{K-1} e^{\beta_{lk0} + \beta_{l1}X_1 + \cdots \beta_{lp}X_p}} \quad \text{for } k = 1,\dotsc,k-1 \\
\text{and}\\
P(Y=K|X=x) &= \frac{1}{1 + \sum_{l=1}^{K-1} e^{\beta_{lk0} + \beta_{l1}X_1 + \cdots \beta_{lp}X_p}} \quad \text{for baseline } Y=K \\
\\
\\
\text{in fact, for }  k = 1,\dotsc,k-1\\
\log \frac{P(Y=k|X=x)}{P(Y=K|X=x)} &= \beta_{Kk0} + \beta_{k1}X_1 + \cdots \beta_{kp}X_p
\end{aligned}
$$

**the last equation allows us to find the logodds between any k class against the baseline**

The decision to treat the Kth class as the baseline is not important. The coeffients estimates will differ between the fitted model based on the choice of baseline. but the fitted values/prediction and the log odds between any pair of classes and other key model outputs remain the same. 

still, the interpretation of the coefficients is tied to the baseline. from the book example, if select seizure as baseline, we can interpret $\beta_{stroke0}$ as the log odds of strike vs seizure, given that $x_1,\cdots,x_p = 0$. And if $X-j$ increases by 1 unit then $ \log \frac{P(Y=stroke|X=x)}{P(Y=seizure|X=x)}$ increases by $e^{\beta_{strokej}}$

(we introduce softmax coding another time)
