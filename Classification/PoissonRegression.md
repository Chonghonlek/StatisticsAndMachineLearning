Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

Our objective in this series was to predict Y - which is assumed to be qualitative. 

We now deal with situations where Y is neither qualitative nor quantitative. Eg Bike hours - the response only takes on non-negative integers or counts in general. 

Before we move on to poisson regression, we analyse the use of linear regression. 

## Transformation of linear regression 

(from textbook example) when linear regression was fitted into the bikeshare data - about 10% of the fitted value were negative!

Also, it is reasonable to suspect that whne the expected value of bikers is small, the variance is small as well. Refer to data below

![Fig 4.14 from G James et al ](../images/Screenshot%202024-07-18%20Fig4.14%20G%20james.png)

There is a major violation in the assumption of the linear model - $Y = \sum X_j\beta_j + \epsilon$ where the error term is assumed to have mean = 0 and constant variance. However the heteroscedascity (assumption of equal variance) of the data calls into question the suitability of the linear regression model. Coupled with response being integer valued, linear regression model may not entirely be satisfactory. 

One way to overcome this is transforming the response so that we have:

$$
\log(Y) = \sum_j X_j \beta_j + \epsilon \iff Y = \exp (\sum_j X_j \beta_j + \epsilon)
$$

This transformation avoids the possibility that Y is negative. 

However, this may not be satisfactory solution. "a one unit increase in $X_j$ is associated with an increase in the mean of log(Y) by $\beta_j$" - is hard to interpret. Also, it cannot be applied in settings where the response can take on a value of 0. 

## Poisson Regression

Recall that in a poisson distribution has a support $k \ge 0$ and 

$$
P(Y=k) = \frac{e^{-\lambda} \lambda^k}{k!}
$$

this is natural to model counts since Y only takes on non-negative integer values. 

Also, $E(Y) = Var(Y) = \lambda$

We expect that $\lambda = E(Y)$ to vary as function of hour, month, season, weather condition, etc. Thus we like to allow the mean to vary as a function of the covariates - $\lambda(X_1,\dotsc,X_k)$. we set

$$
\log(\lambda(X_1,\dotsc,X_k)) = \beta_0 + \beta_1X_1 + \cdots + \beta_k X_k \iff \lambda(X_1,\dotsc,X_k) = e^{\beta_0 + \sum \beta_iX_i}
$$

Here, we are setting the **log of the mean of Y or $\lambda$** to be linear in $X_1 ,\cdots, X_k$, which allows lambda to take on non-negative values. 

To estimate the poisson regression coefficients we use MLE apporach - calculate the total probability of observing all of the data, i.e. the joint probability distribution of all observed data points where we assume indepedence of data points.

$$
\begin{aligned}
    L(\beta_0,\cdots,\beta_k) &= \prod_i P(Y = y_i)\\
    &= \prod_i \frac{e^{-\lambda(x_i)}\lambda(x_i)^{y_i}}{y_i !} \quad \text{, where }\lambda(x_i) =  e^{\beta_0 + \sum \beta_iX_i}
\end{aligned}
$$

Method similar to logistic regression

To interpret the coefficients of poisson regression model, a change in $X_j$ leads to a corresponding change in $\lambda = E(Y)$ by a factor of $\exp(\beta_j)$

Given that $E(Y) = Var(Y) = \lambda$, modelling bike usage using poisson regression,  we implicitly assume that the mean of bike usage = variance of bike usage in that hour. thus, this handles the mean-variance relationship better than lineae regression. 

## Generalized Linear models in greater generality 

1. Each approach uses predictors $X_1 , \cdots, X_k$ to predict a response. We assume , conditional on $X_1 , \cdots, X_k$ that Y belongs to a certain family of distributions. For linear regression, we assume Y follows a normal distribution. For logistic regression, Y is assumed to follow benoulli distribution. And, poisson regression - self explanatory. 

2. each approach models the mean of Y as a function of predictors. 
- in linear regression we have

$$
E(Y|X_1,\dotsc,X_k) = \beta_0 + \beta_1X_1 + \cdots + \beta_k X_k
$$

- in logistic regression, we have 

$$
E(Y|X_1,\dotsc,X_k) = p(X) = \frac{e^{\beta_0 + \beta_1X_1 + \cdots \beta_kX_k}}{1 +e^{\beta_0 + \beta_1X+ \cdots \beta_kX_k} }
$$

- in poisson regression, we have 

$$
E(Y|X_1,\dotsc,X_k) = \lambda(X_1,\dotsc,X_k) = e^{\beta_0 + \beta_1X_1 + \cdots + \beta_k X_k}
$$

---

Note that these distributions belongs to the exponential family. Some other members include the exponential, gamma and negative binomial distirbution. 

In general, we can perform regression by modeling the response Y as coming from a particular member from the exponential family. Then transforming the mean of response so that the mean is a linear function of the predictors. 

In fact, any regression approach that follows this recipe is known as a generalised linear model (GLM)