

Each leg of a patient's journey through the hospital system can be characterised by 

1. the length of stay $\ell_i$ in a given hospital $
2. the readmission time between hospital admissions. $\tau$

In a naive static model, we model $\ell_i \sim \text{Exp}(\lambda_i)$ for some given hospital $i$, and $\tau = 0$.

We can improve on this static network model by having:

- $\ell_i \sim \text{Exp}(\lambda_i)$ as above
<!-- - $\tau_{ij} \sim \begin{cases}\text{Delta}(0) & \text{with probability } p_{ij} \\ \text{Exp}(\lambda^{(H)}_{ij}) & \text{with probability } (1-p_{ij})\end{cases}$

$$\begin{aligned}
\left(\lambda^{(H)}_{ij}\right)^{-1} 
&= \int_\Omega^\infty f_{ij}(\tau) \tau d\tau\\ 
&\approx \frac{1}{|\{\tau^{(n)}_{ij} : \tau^{(n)}_{ij} > \Omega\}|}\sum_{\tau^{(n)}_{ij} > \Omega} \tau^{(n)}_{ij}
\end{aligned}
$$ -->


- $F(\tau_{ij}) = 1 - (1-p_{ij}) e^{-\lambda_{ij}^{(H)} t}$

where $\lambda_{ij}^{(H)}$ is the empirical rate of indirect movement from $i$ to $j$ and $p_{ij}$ is the probability that a movement from $i$ to $j$ is direct.
$\lambda^{(H)}_{ij}$ can be computed as the empirical mean of the readmission times above some threshold $\Omega$:

$$
\left(\lambda^{(H)}_{ij}\right)^{-1} 
= \frac{1}{|\{\tau^{(n)}_{ij} : \tau^{(n)}_{ij} > \Omega\}|}\sum_{\tau^{(n)}_{ij} > \Omega} \tau^{(n)}_{ij}
$$

where $\tau^{(n)}_{ij}$ is an observed readmission time where an individual travels from $i$ to $j$. 

We can also compute
$$p_{ij} = \frac{|\{\tau^{(n)}_{ij} : \tau^{(n)}_{ij} < \Omega\}|}{|\{\tau^{(n)}_{ij}\}|}$$

This makes the modelled readmission distribution more reflective of the actual readmission distribution (i.e. non-zero). This also introduces paths into the movement of patients that take a non-zero amount of time to traverse, which will raise the (expected) hitting time.

If we use a series of static networks (_snapshots_ of the temporal network), we can allow variation in $\ell_{i}$ and $\tau_{ij}$ over time. This variation, particularly when ??, causes a non-trivial distribution for $\tau_{ij}$.



We can allow variation in $\ell, \tau$ over time by introducing snapshots. 


We can then capture the correlations between stuff with a full temporal network.


