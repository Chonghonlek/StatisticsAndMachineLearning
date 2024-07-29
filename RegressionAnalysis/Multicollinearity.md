Notes taken from **Douglas C M et al, Introduction to Linear Regression Analysis**

# Multicollinearity

When there near-linear dependencies among the regressors , the problem of multicollinearity is said to exist.

Recall that exact dependency will cause $X^TX$ to be singular. and high dependecy causes $X^TX$ to be **near singular(ill-conditioned)**, especially causing $(W^TW)^{-1}$ to have large diagonal elements (VIFs). This causes $\beta$ to be estimated wrongly. (impact)

when there more than 2 regressor variables, diagonal elements of $(W^TW)^{-1}$ is

$$
C_{jj} = \frac{1}{1-R^2_{j}}
$$

where $R^2_j$ is the coefficient of multiple determinatipn from the regression of $x_j$ on and the remaining p-1 regressors. 
- strong multicollinearity will causes high $R^2_j$ and also $Var(\hat{\beta_j}) = C_jj \sigma^2$ to be very large. 

---
Recall: eigen values $(\lambda_1, \dotsc, \lambda_p)$ of A are roots of $|A-\lambda I_p| = 0$ and that eigenvector t is such that: $At = \lambda t$

we examine the decomposition of symmetric positive definite A:

1) $A = T\Lambda T^{-1}$ where $T = (t_1,\cdots,t_p)$ are orthogonal eigenvectors
2) $\Lambda = \begin{pmatrix} \lambda_1 &\cdots &0 \\ &\vdots \\ 0 &\cdots &\lambda_p\\ \end{pmatrix}$ this is diagonal martix of positive eigenvalues of A
3) $T^T = T^{-1}, T^TT = I_p$
4) $trace(A) = \sum_i \lambda_i$
5) $trace(A^{-1}) = \sum_i \frac{1}{\lambda_i}$

We look at the relationship between eigenvalues of $X'X$ and multicollinearity.

$$
\begin{aligned}
    L_1^2 &= (\hat{\beta} - \beta)^T (\hat{\beta} - \beta) \\
    E(L_1^2) &= \sum_j E [(\hat{\beta_j} - \beta_j)^2] = \sum_j Var(\hat{\beta_j}) \\
    &= \sigma^2 \times \text{ sum of diagonal entries of } (X^TX)^{-1} \\
    &= \sigma^2 \times trace((X^TX)^{-1}) \\
    &= \sigma^2 \sum \frac{1}{\lambda_j}
\end{aligned}
$$

From above, we can see that small eigenvalues of $(X^TX)$ will result in poorly estimated $\beta \iff  E(L_1^2)$ is very large

Note that :

$$
\begin{aligned}
    X^TXt_j &= \lambda_jt_j \\
    \iff X^T(t_{1j}X_1 + \cdots + T_{pj}X_p) &= \lambda_j
    \begin{pmatrix}
        t_{1j} \\ \vdots \\ t_{pj}
    \end{pmatrix} \\
    \iff \text{if } \lambda_j \approx 0 \implies X^T(t_{1j}X_1 + \cdots + T_{pj}X_p) &\approx 0 \\
    \implies Xt_j &\approx 0 \iff \text{dependency among regressors}
\end{aligned}
$$

Recall:
Sources of multicollinearity comes from:
1) data collection from only a subspace of the regressors.
2) Constaints on model - (correlated regressors must included the model)
3) Non-centering of regressor variables
4) Overdefined model (more regressor than observations)
   
## Diagnostics

3 ways to detect multicollinearity:
1) looking at the correlaton matrix -  the off diagonal elements $r_{ij}$ in $(X^TX)$. 
   - but if more than 2 regressor may not show that the pairwise correlation is large. e.g. Linear dependecny of 4 regressors cannot be detected by just looking at regressor variables
2) Variance Inflation factors (VIFs) - we look at $C=(X^TX)^{-1}$ and $VIF_j = \frac{1}{1-R^2_j}$. if $ VIF_j >> 1$, imply high correlation.
- VIF measures how much the variance of $\hat{\beta_j}$ is affected by other regressors
- $VIF \ge 2.5$ signal evidence of multicollinearity
3) Eigenvalue analysis of $X^TX$ - where small eigenvalues indicate evidence of multicollinearity
   - we can look at the **condition number**,$\kappa$ of $X^TX$, which we define as 
$$
\kappa = \frac{\lambda_{max}}{\lambda_{min}}, \quad \lambda_{max} \text{is max of eigenvalue and } \lambda_{min}  \text{is min of eigenvalue}
$$
   - if $\kappa \lt 100$, there is no serious problem.
   - if $100 \le \kappa \le 1000$, there is moderate -serious collinearity. 
   - anything above is severe

in r code:

```r
dec = eigen(XPX)
T = dec$vector
D = dec$values
con.num = max(D)/max(i)
con.index = max(D)/D
```

# Solutions to MultiCollinearity

1) Regressor varaible elimination - here we eliminate one of the variables that is highly correlated. This method is not satisfactory if the variable dropped has significant explanatory power of the response variable
2) Respecification of model  - suppose $x_1,x_2,x_3$ are highly linearly dependent. We can introduce a new regressor variable that is a function of $x_1,x_2,x_3$ and yet preserves their informatiion. then using this  new variable will eleimaite the multicollinearity problem. e.g. x = x1*x2 , x1 = breadth and x2 = length
3) ridge regression
4) Principle component regression

## Ridge Regression

Consider an example where, we have A such that $A^{-1}$ does not exist. We can apply $A + kI_p$ such that $(A+kI_p)^{-1}$ exist. this is called strengthening the ridge (diagonals) of the matrix.

In ridge regression, we have 

$$
(X'X + kI)\hat{\beta_R} = X'y \iff \hat{\beta_R}  = (X'X + kI)^{-1} X'y
$$

Properties:
1) ridge estimator is a linear transformation of least square estimator $\hat{\beta_R} = (X'X + kI)^{-1} X'y = (X'X + kI)^{-1} X'X\hat{\beta} = Z_k\hat{\beta}$
2) the ridge estimator $\hat{\beta_R}$ is a biased estimator of $\beta$. i.e. $E(\hat{\beta_R}) = Z_k\beta \ne \beta$ unless k = 0
3) Varaince-covariance matrix of $\hat{\beta_R}$:

$$
Var(\hat{\beta_R})  = \sigma^2 (X'X + kI)^{-1}X^TX (X'X + kI)^{-1}
$$

4) $MSE(\beta_R)$ - here is we look at mean square error. not $MS_{res}$
- Recall that $MSE(\hat{\beta}) = E[(\hat{\beta}-\beta^2)] = Var(\hat{\beta}) + (E(\hat{\beta}) - \beta)^2$ where $(E(\hat{\beta}) - \beta)^2$ is known as the bias

$$
\begin{aligned}
    MSE(\beta_R) &= E[(\hat{\beta_R} - \beta)^T(\hat{\beta_R} - \beta)] \\
    &= trace(Var(\hat{\beta_R})) + (E(\hat{\beta_R}) - \beta)^T(E(\hat{\beta_R}) - \beta) \\
    &= \sum_j Var(\hat{\beta_{Rj}}) + \sum_j [E(\hat{\beta_{Rj}}) - \beta_j]^2 \\
    &= \sigma^2 \sum_{j=1}^p \frac{\lambda_j}{(\lambda_j + k)^2} + k^2 \beta^T(X^TX + KI)^{-2}\beta
\end{aligned}
$$

- As k increases, the total variance decreases but the total bias increases.

5) $SS_{res} = (y-X\hat{\beta_R})^T(y-X\hat{\beta_R}) = (y-X\hat{\beta})^T(y-X\hat{\beta}) + (\hat{\beta_R} - \hat{\beta})^TX^TX(\hat{\beta_R} - \hat{\beta})$
- As k increases, $SS_{res}$ increases because the bias of the ridge estimate increases, hence $R^2$ will decrease. 
- It is not a good idea to use a large k

6) The ridge regression can be computed by using the ordinary least squares approach (proof not shown)

$$
y_A
\begin{pmatrix}
    y \\ 0_p
\end{pmatrix}, X_A =
\begin{pmatrix}
    X \\ \sqrt{k}I_p
\end{pmatrix}, \hat{\beta_R} =
(X_A^TX_A)^{-1}X^T_Ay_A = (X^TX + kI_p)^{-1}X^Ty
$$

### How to choose a biasing constant k?

- we make a ridge trace which is a plot of elements of the** ridge estimate $\hat{\beta_R}$ verses k for k in the interval (0,1)**
- As k increases, the ridge estimates will vary but stabalise for larger values of k
- choose a reasonably small value of k at which the ridge estimates are stable

![fig9.5 from D Montgomery et al](../images/Screenshot%202024-07-27%20Fig9.5%20D%20Montgomery.png)

(refer to notes for R code)

### Summary of Ridge Regression

- When the $X^TX$ matrix is ill-conditioned because of multi-collinearity, ridge regression uses a simple idea of imporving the condition of the X'X matrix. It strengthens the ridge of $X^TX$ so that it is no longer ill conditioned:

$$
(X^TX + kI)\hat{\beta_R} = X^Ty, \text{ where }k \ge 0
$$

## Principle component regression

In this approach of reducing multicollinearity, we instead remove the dependency among the regressor variables. Hence this results in a smaller $X^TX$ that is no longer ill-conditioned

Recall from above,  $X^TX$ is symmetric and thus orthogonal and diagonalizable matrix.  

$$
X^TX = T\Lambda T^{-1} =T\Lambda T^T, \text{where }T = (t_1,\cdots,t_p) \text{ are orthogonal eigenvectors}
$$

and that $\Lambda = \begin{pmatrix} \lambda_1 &\cdots &0 \\ &\vdots \\ 0 &\cdots &\lambda_p\\ \end{pmatrix}$ this is diagonal martix of positive eigenvalues of A

Consider $y =XTT^T\beta + \epsilon =  Z\alpha + \epsilon$ where (note: $TT^T = I_p$)

$$
Z=XT, \quad \alpha = T^T\beta, \quad Z^TZ = TX^TXT = \Lambda 
$$

The columns of $Z$ define a new set of orthogonal regressors known as principle components. $Z = \begin{pmatrix} Z_1 Z_2 \cdots Z_p \end{pmatrix}$

The least squares estismator of $\hat{\alpha}$ is

$$
\hat{\alpha} = (Z^TZ)^{-1}Z^Ty = \Lambda^{-1}Z^Ty
$$

The convariance matrix of $\hat{\alpha}$

$$
Var(\hat{\alpha}) = \sigma^2(Z^TZ)^{-1} = \sigma^2 \Lambda^{-1}
$$

we have that $Var({\hat{\alpha_j}}) = \frac{1}{\lambda_j}$, where small eigenvalue will result in larger variance in the corresponding orthogonal regression coefficient

- even when Z is orthogonal, does not guarantee a small eigenvalue.
- Since $Var(\hat{\beta}) = T\Lambda^{-1}T^T\sigma^2$ this implies $Var(\hat{\beta_j}) = \hat{\sigma_2}(\sum_i t_{ji}^2 / \lambda_i)$ and that the variance of $\hat{\beta_j}$ is a linear combination of the reciprocals of the eigenvalues
- in r: 

```r
e = eigen(XPX)
T = e$vectors
p = e$values
```

### how it works

1. Assume the regressor variable are arranged in order of decreasing eigenvalues $\lambda_1 > \cdots >\lambda_p$
2. set up $y =\alpha_1z_1 + \cdots +\alpha_{p-s}z_{p-s} + \cdots + \alpha_pz_p + \epsilon$
3. Suppose last s eigenvalues $(\lambda_{p-s+1},\cdots,\lambda_p)$ are near zero, i.e. $z_{p-s+1} = Xt_{p-s+1} = 0$ since Z will be ill conditioned as last s columns are near 0
4. set the last s cols $(z_{p-S+1},\cdots,z_{p})$ as zero vector and the corresponding regressor coefficients as 0
   - then we have $y  = \alpha_1z_1 + \cdots +\alpha_{p-s}z_{p-s} + \epsilon =$ 
   - here is where remove the dependecy among regressor variables
5. in this new  $Z_* = \begin{pmatrix} Z_1 Z_2 \cdots Z_{p-s} \end{pmatrix}$ is no longer ill conditioned
6. we can then estimate: $\hat{\alpha_{pc}} = \begin{pmatrix} \hat{\alpha_{1}}\\ \hat{\alpha_{2}} \\ \vdots \\ \hat{\alpha_{p-s}}  \\ 0 \end{pmatrix}$, where we had set the last s cols to 0
7. Principle Component estimates then can be obtained by $\hat{\beta_{pc}} = T\hat{\alpha_{pc}}$
8. This is infact a more compact model

Note that there is subjectipon to decide how many principle components(PC) to remove. The more near-zero PC removed, the more stable your system. 

Scaling of regressor variables also not required.