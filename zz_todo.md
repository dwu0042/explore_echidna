5 Dec 2024
==========

plot the empirical length of stay distribution, see if it agrees with Exp(gamma)

- [10/12] Need to access VALT/CVDL for this. We shold be able to get this from the cleaned_csvs (should I use the new data? - probably not, since the networks were created using the old data)
- [12/12] Looks like we already did this. 
    - It was done by aggregating over campus (so that we could compute the probability of final stay)
    - The graph doesn't build if we send through the raw info through to seaborn
        - maybe too many entries?
        - I've fixed on a np.histogram + plt.plot pipeline (plt.bar also crashes similarly)

See if we can produce an analytical distribution of hitting time for the naive static network
  - I know we can do expected hitting time...

  - [11/12] We can compute the expected hitting time and plot that: the distribution wil be different, but I'll need to look up properties of the sample of the mean etc
  - [11/12] The problem here is that we have 30 agents moving around in the simulations, whereas the analytical expected hitting time is a single agent jumping around. We'll need to correct for this, but how?...
  - [11/12] I also need to commit the hitting time analysis to code (I know we have the script and the lib, but I need to explicitly construct the artifact generator).

Analyse the properties of the ndoes that are not reached:
  - in the static, are these small nodes?
  - can we lookup nodes that are not reached in static - do they have a guaranteed or hig prob path in the snapshot?
  - 

10 Dec 2024
===========

The holiday project should be refactoring stuff into a ploomber-ish pipeline.
We have some starting points, but not all.
    - One thing I'll have to work out is how we make the pipeline history the artifacts.
        - We want to keep old sims for example.
            - We should also copy over the old sims to backup
            - Can we attribute the old sims as well?

13 Jan 2025
===========

Holiday project: gillespie's direct method in Go [still incorrect/incomplete]
After ANZIAM project: prefect pipelines

Also, over the holidays, I think I also determined that p = 0.5 isn't the right thing to compute (since that returns the median)

Things learnt today:
- The quantity(ies) p(t) is a stochastic process.
- The quantity $D_A$, the hitting time for subset of states $A$, _is_ a random variable. This has a distribution.
- We compute $h_A$ the expectation of $D_A$ through a linear system.
- The expectation of $D_A$ is computed through the following integral:
  $$\mathbb{E}[D_A] = \int_\Omega f(t) t dt$$
  where $\Omega$ is the support of $D_A$ (usually $[0, \infty)$), and $t$ represents $D_A$.
- The survival function $S_D(t)$ is important here. The survival function is the probability that $D_A > t$. 
- Firstly, we can integrate the survival function to get the expectation too:
  $$\mathbb{E}[D_A] = \int_\Omega S_D(t) dt$$
- Then we can also use the CDF trick to get expectation of $M = \min{(D_A^{(1)}, D_A^{(2)}, \dots D_A^{(N)})}$
  $$ \mathbb{E}[M] = \int_\Omega S_M(t) dt = \int_\Omega S_D(t)^N dt$$
- Apparently, we can also determine the survival function using the Kolmogorov backwards equations:
  $$\frac{d}{dt} u(t) = Q^*u(t)$$
  where the solution of $u(t)$ is $S(t)$, when $u(0) = 1$ for all indices not in $A$ and $0$ otherwise, and we have $Q^*[A,:] = \vec{0}$.
  Alternatively, we can partition $Q$ based on the indices of $A$, and have instead
  $$\frac{d}{dt} u(t) = Q_{A^cA^c} u(t)$$
  where $u(t)$ is now the survival probability of states $A^c$, the complement of $A$ in $I$. This is just a reduction of the previous equation for states that are known (i.e. $A$)
- If $D_A$ is exponentially distributed, i.e $S_D(t) \sim \exp(-t/h_A)$ for some mean (expected) hitting time $h_A$, then the solution simplifies to $\mathbb{E}[M] = h_A / N$ due to the linearity of the exponential distribution -> [14/01] this doesn't work for us

- 

15 Jan 2025
===========

we now have a straightforward strategy for computing the expected hitting times
  - Set up the integral
    - $$ \int_0^\infty \exp(-t) (\exp(t) u(t)) dt$$
    - where $u(t)$ is the survival function
  - We can use Gauss-Laguerre quadrature to solve this form of integral
  - This involves computing M locations to evaluate u(t) at, and weighted sum with M weights
  - We can compute $u(t)$ numerically using numerical integration
    - This should be relatively straightforward (a 400-D linear ODE)

22 Jan 2025
===========

Gauss-Laguerre does not converge well when the decay rate is slow.
We can fix this with a linear rescaling of time, should be able to detect via the eigenvalues of the reduced Q matrix

- Roughly, for eigenvalues $-\lambda_i$ and eigenvectors $v_i$, the characteristic time scales as 
  $$\max \Bigg[\frac{\log(v_i) - \log(q)}{\lambda_i}\Bigg]$$
  where $q$ is some quantile.
  For now, we can choose $q$ arbitrary, but we might want to have a guess at what $q$ results in a stable convergence of the G-Lgg integration

24 Jan 2025
===========

We can perform an explicit expansion of the exact solution, and use decomposition to get the exact coefs and function params.
This allows us to get an exact solution in time prop to the number of chains factorial (? need to check scaling) + size fo the problem (for the eigen problem)

[26 Jan] Looks like scaling is a problem here... (50 x 30 is not really competing in finite time)

26 Jan 2025
===========

Did a bit of experimenting with scaling.
We actually see that $v_i$ has little to no impact on the effect of scaling when we do

$$\int_o^\infty v_i e^{-\lambda_i t} = s\int_o^\infty v_i e^{-\lambda_i st}$$

since the integral evaluates to $v_i / \lambda$. I think I may have made an error when deriving the expression, and didn't cancel a factor of $v_i$ when dealing with the RHS boundary (which is probably something like $q \times v_i$)

Any way, it looks like choosing $q \approx 0.05$ works pretty well on a single term. Checking for multiple terms now.

[Additional thought] is that the exponentiation by $N$ does change the characteristic time. The problem here is determining a good characteristic time to choose. I guess since $N$ will multiply with the exponential, that simply choosing the standrad characteristic time $\times N$ will work...?

One thing we need to consider is that at long time, the smallest decay rate will dominate, regardless of choice of $v_i$.

[28/01] We have learnt that:
- as $v_i$ increases, the window does not really change.
- as $1/\lambda$ increases, the window of appropriate $s$ moves to the right. 
- as the precision increases, the width of the window increases, centred on the centre of the window.
- as we include more terms, the window decreases in size. Using q=0.05, and log(q)/lambda doesn't seem to work very well any more.
- 