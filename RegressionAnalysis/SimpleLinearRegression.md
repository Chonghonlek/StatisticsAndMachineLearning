Notes taken from **Douglas C M et al, Introduction to Linear Regression Analysis**

# Simple linear regression

Assuming y has a linear relationship with its predictors, the simple linear regression model takes the form $y = \Beta_0 + \Beta_1x + \epsilon$

In this model, we further assume that $\epsilon$ has mean and variance of 0 and $\sigma^2$ respectively. Which implies that $E(y|x) = E(\Beta_0 + \Beta_1x + \epsilon) = \Beta_0 + \Beta_1x$ and that $Var(y|x) = Var(\Beta_0 + \Beta_1x + \epsilon) = \sigma^2$

Additionally we assume that the errors are uncorrelated, which imply repsonses are also uncorrelated. 

The parameters $\Beta_0$ and $\Beta_1$ are the regression coefficients. They are unknown and will be estimated using sample data. 

## Estimation of $\Beta_0$ and $\Beta_1$

we set the least sqaures criterion as $S(\Beta_0,\Beta_1) = \sum_{i=1}^n (y_i - \Beta_0 - \Beta_1x_i) = \sum_{i=1}^n \epsilon_i^2$, then the following equations must be satisfied to attain the **least squares estimator** of $\Beta_0$ and $\Beta_1$

$$
\begin{align}
\frac{\partial S}{\partial \Beta_0} = -2 \sum_{i=1}^n (y_i - \hat{\Beta_0} - \hat{\Beta_1 x_i}) =^{\text{set to}} 0\\
\frac{\partial S}{\partial \Beta_1} = -2 \sum_{i=1}^n (y_i - \hat{\Beta_0} - \hat{\Beta_1 x_i})x_i =^{\text{set to}} 0
\end{align}
$$

$$
\text{From (1),} \quad n\hat{\Beta_0} + \hat{\Beta_1} \sum x_i = \sum y_i \\
\text{From (2),} \quad \hat{\Beta_0}\sum x_i + \hat{\Beta_1} \sum x_i^2 = \sum y_i x_i
$$

<details>

<summary>Solving,</summary>

$$
\begin{aligned}
n\hat{\Beta_0} + \hat{\Beta_1} \sum x_i &= \sum y_i \\
n\hat{\Beta_0} + n \bar{x}\hat{\Beta_1} &= n \bar{y} \\
\hat{\Beta_0} &= \bar{y} - \hat{\Beta_1}\bar{x} \tag{coefficient of $\Beta_0$} \\
\end{aligned}
$$

$$
\begin{aligned}
\text{sub $\Beta_0$ into (2),} \quad \sum y_i x_i - (\bar{y} - \hat{\Beta_1}\bar{x})\sum x_i - \hat{\Beta_1} \sum x_i^2 &=0 \\
\sum y_i x_i - \bar{y}\sum x_i - \hat{\Beta_1}\bar{x}\sum x_i - \hat{\Beta_1} \sum x_i^2 &=0 \\
\hat{\Beta_1} (\sum x_i^2 -\bar{x}\sum x_i) = \sum y_i x_i - \bar{y}\sum x_i
\\
\hat{\Beta_1} = \frac{\sum y_i x_i - \frac{\sum y_i\sum x_i}{n}}{\sum x_i^2 - \frac{(\sum x_i)^2}{n}} \tag{coefficient of $\Beta_1$}
\end{aligned}
$$

</details>

$$
\hat{\Beta_0} = \bar{y} - \hat{\Beta_1}\bar{x}  \\
\implies \text{rearranging,} \quad \bar{y} = \hat{\Beta_0} + \hat{\Beta_1}\bar{x} \implies \text{least square line pass through mean of x and y} \\
$$

and 

$$
\begin{aligned}
\hat{\Beta_1} &= \frac{\sum y_i x_i - \frac{\sum y_i\sum x_i}{n}}{\sum x_i^2 - \frac{(\sum x_i)^2}{n}} \\
\text{in fact,} \quad &=\frac{\sum y_i(x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2} = \frac{\sum y_i x_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2} = \frac{\sum (x_i -\bar{x})(y_i -\bar{y})}{\sum (x_i -\bar{x}) ^2}  \\
&\because \sum (x_iy_i - \bar{x}y_i - x_i \bar{y} + \bar{x}\bar{y}) = \sum (x_iy_i) - n\bar{x}\bar{y} \\
& = \frac{S_{XY}}{S_{XX}} = \frac{\text{Sample Cov}}{\text{Sample Var}}
\end{aligned}
$$

## Properties of the Least square estimators

We begin with a simple lemma (Lemma 1.1): Let $y_i,\dotsc,y_n$ be n mutually independent normal variables where $Y_i \sim N(\mu_i,\sigma_i^2)$ for $i = 1,\dotsc,n$, then $\sum C_i Y_i \sim N(C_i \mu_i, \sum C_i^2 \sigma_i^2)$ where $\mu_i = \Beta_0 + \Beta_1x_i$

### $\hat{\Beta_1}$

$$
\hat{\Beta_1} = \frac{\sum y_i(x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2} \sim N(\Beta_1, \frac{\sigma^2}{S_{XX}})
$$

<details>
<summary>Proof</summary>

$$
\begin{aligned}
C_i &= \frac{(x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2} \\
\\
E(\hat{\Beta_1}) &= \sum \frac{ (x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2} (\Beta_0 + \Beta_1x_i) \\
& = \Beta_0 \sum (\frac{ (x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2}) + \Beta_1 \sum (\frac{ x_i(x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2}) \\
& = 0 + \Beta_1 \sum (\frac{ (x_i^2 -x_i\bar{x})}{\sum (x_i -\bar{x}) ^2}) \because \sum (x_i - \bar{x}) = 0\\
& = \Beta_1 (\frac{ \sum x_i^2 - \sum x_i\bar{x}}{\sum (x_i -\bar{x}) ^2}) = \Beta_1 (\frac{ \sum x_i^2 - n\bar{x}^2}{\sum x_i^2 - n\bar{x}^2}) = \Beta_1 (1) \\
& = \Beta_1
\\
Var(\hat{\Beta_1}) = \sum \frac{ (x_i -\bar{x})^2 \sigma^2}{(\sum (x_i -\bar{x}) ^2)^2}  &= \frac{\sigma^2}{\sum (x_i -\bar{x}) ^2} = \frac{\sigma^2}{S_{XX}}
\end{aligned}
$$

</details>

### $\hat{\Beta_0}$

$$
\hat{\Beta_0} = \bar{y} - \hat{\Beta_1}\bar{x} \sim N(\Beta_0,(\frac{1}{n} + \frac{\bar{x}^2}{S_{XX}})\sigma^2)
$$

<details>
<summary>Proof</summary>

$$
\begin{aligned}
E(\hat{\Beta_0} ) &= E(\bar{y} - \hat{\Beta_1}\bar{x}) \\
&= \frac{1}{n} \sum(E(y_i)) - \frac{1}{n} \sum E(\hat{\Beta_1} x_i) \\
&= \frac{1}{n} \sum (\Beta_0 + \Beta_1x_i) - \frac{1}{n} \sum \Beta_1 x_i \\
& = \Beta_0
\end{aligned}
$$

Lemma 1.2: $Cov(\bar{y},\hat{\Beta_1}) = 0$

$$
\begin{aligned}
Cov(\bar{y},\hat{\Beta_1}) &= Cov(\frac{1}{n}\sum y_i,\sum C_i y_i), \quad C_i = \frac{(x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2} \\
&= \frac{1}{n} \sum_i Cov(y_i,C_iy_i) + \frac{1}{n} \sum\sum_{i \neq j} Cov(y_i,C_j,yj) \\
&= \frac{1}{n} \sum_i C_i Cov(y_i,y_i) + 0 \quad \because \text{$y_i$ are indepedent} \\
&= \frac{1}{n} \sum_i C_i Var(y_i) = \frac{\sigma^2}{n} \sum_i C_i \\
&= 0 \quad \because \sum \frac{(x_i -\bar{x})}{\sum (x_i -\bar{x}) ^2} = 0
\end{aligned}
$$

We use Lemma 1.2 to proof the variance of $\hat{\Beta_0}$

$$
\begin{aligned}
Var(\hat{\Beta_0}) &= Var(\bar{y} - \hat{\Beta_1}\bar{x}) = Var(\bar{y}) + (\bar{x})^2 Var(\hat{\Beta_1}) + 2 \bar{x}Cov(\bar{y},\hat{\Beta_1}) \\
& = \frac{\sigma^2}{n} + (\bar{x})^2\frac{\sigma^2}{\sum (x_i -\bar{x}) ^2} + 0 \quad \text{from Lemma 1.2} \\
& = \sigma^2 (\frac{1}{n} + \frac{\bar{x}}{S_{XX}})
\end{aligned}
$$

</details> <br>

The results show that the parameters $\hat{\Beta_0}$ and $\hat{\Beta_1}$  are unbiased estimators of the model parameters $\Beta_0$ and $\Beta_1$

Gauss Markov theorem states that for the regression model, the least squares estimators are unbiased and have minimum variance. 

### Other useful properties of the least square fit

1. Sum of the residuals is always zero : $ \sum (y_i - \hat{y_i}) = \sum e_i = 0$
2. Sum of observed values equal sum of fittec values: $\sum y_i  = \sum \hat{y_i}$
3. Least squares regression line passes through the centroid $(\bar{x},\bar{y})$ : $\bar{y} = \hat{\Beta_0} + \hat{\Beta_1}\bar{x}$ (shown above)
4. Sum of residuals weighted by value of regressor is 0 : $\sum x_i e_i = 0$
5. Sum of residuals weighted by fitted value is 0 : $\sum \hat{y_i} e_i = 0$

## Estimating $\sigma^2$

We define the residual sum of square as $SS_{res} = \sum e_i^2 = \sum (y_i - \hat{y_i})^2$

$$
\begin{aligned}
SS_{res} &= \sum (y_i - \hat{y_i})^2 = \sum (y_i - \hat{\Beta_0} + \hat{\Beta_1}x_i)^2 \\
& = \sum (y_i - (\bar{y} - \hat{\Beta_1}\bar{x})  + \hat{\Beta_1}x_i)^2 = \sum ( (y_i - \bar{y}) - \hat{\Beta_1}(x_i - \bar{x}))^2 \\
&= \sum (y_i - \bar{y})^2 + \hat{\Beta_1}^2\sum(x_i - \bar{x})^2 - 2 \hat{\Beta_1} \sum (x_i -\bar{x})(y_i -\bar{y}) \\
&= \sum (y_i - \bar{y})^2 + \hat{\Beta_1}^2 S_{XX} -  2\hat{\Beta_1}S_{XY} \\
&= \sum (y_i - \bar{y})^2 + \frac{S_{XY}^2}{S_{XX}}  -  2\hat{\Beta_1}S_{XY}\\
&= SS_T -  \hat{\Beta_1}S_{XY} \\
&= SS_T - SS_R \\
\\
\implies  SS_T &= SS_{res} + SS_{R}\\
\implies \sum (y_i - \bar{y})^2 &= \sum (y_i - \hat{y_i})^2 + \sum (\hat{y_i} - \bar{y})^2
\end{aligned}
$$

Here, $SS_T$ is the total sum of square which describes the variation of the data. $SS_R$ is the regression sum of square which describes the variation explained by the regressor

The residual sum of squaees has n-2 degree of freedom, because 2 degree of freedom are associated with estimates $\hat{\Beta_0}$ and $\hat{\Beta_1}$ involved in obtaining $\hat{y}$

In fact, $E(SS_{res}) = \sigma^2 E(\frac{1}{\sigma^2} \sum(y_i - \hat{y_i})^2) = (n-2) \sigma^2 \quad \because \frac{\sum (y_i - \hat{\Beta_0} + \hat{\Beta_1}x_i)^2}{\sigma^2} \sim \chi^2 (n-2)$. Refer to Douglas C M et al for details of proof

So to get an unbiased estimator of $\sigma^2$, we let **Residual Mean Square**, $\hat{\sigma^2} =MS_{res} = \frac{SS_{res}}{n-2}$. We verify that $MS_{res}$ is unbiased: $\frac{(n-2)MS_{res}}{\sigma^2} \sim \chi^2 (n-2) \implies E(MS_{res}) = \frac{\sigma^2}{n-2} E(\frac{(n-2)MS_{res}}{\sigma^2}) = \frac{\sigma^2}{n-2} (n-2) = \sigma^2$

The square root of $\hat{\sigma^2}$ is known as the **Standard error of regression**. We say that $\hat{\sigma^2}$ is a model depedent estimate of $\sigma^2$ as it is computed from the model residuals. 

$$
\hat{\sigma^2} =MS_{res} = \frac{SS_{res}}{n-2}
$$
