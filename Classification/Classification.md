Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

## Classification

Compared to linear regression, we are trying to predict qualitative response. 

Linear regression is not suitable as it cannot accomodate a qualtitative response with more than 2 cases eg

$$
Y =
\begin{cases}
1 &\text{if a}\\
2 &\text{if b}\\
3 &\text{if c}\\
\end{cases} 
$$

It also does not provide meaning estimates of $P(Y|X)$ even with just 2 classes. Below, we see that some estimates under linear regression go beyond $[0,1]$

![figure 4.2 Taken from Gareth J et al](../images/Screenshot%202024-05-29%20Figure4.2%20G%20James.png)

