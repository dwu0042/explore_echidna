# Deriving the mean hitting time for arbitrary number of starting agents

Let us have a Markov chain $\mathscr{M}$  
We can compute, for one agent, the associated Q-matrix $Q$, and we can write the backward equation as 

$$\frac{d}{dt} p(t) = Q p(t)$$

i.e. the probability distrubtion $p(t)$ of the chain being in each state at time $t$ is dictated by the equation:

$$p(t) = \text{Exp}(Qt)p_0$$

where $p_0$ is the initial probability distribution at $t=0$, and $\text{Exp}$ is the matrix exponential.

This also gives the CDF

$$P(t) = Q^{-1} \text{Exp}(Qt) p_0 - Q^{-1} p_0$$

For $N$ agents, we can write the cumulative probability distribution as

$$\Phi(t) = 1 - (1 - P(t))^N$$

using arguments where we treat the $N$ different agents as independent, and use the CDF trick to generate the appropriate combined CDF.

Then the pdf is $\frac{d}{dt}\Phi(t)$:

$$\phi(t) = N(1 - p_0 Q \text{Exp}(Qt))^{N-1} p_0\text{Exp}(Qt)$$

We want to compute the mean hitting time, which is

$$m = \int_0^\infty t \phi(t) dt$$

Potential problems here is that $p, Q$ etc. are vector-valued, which might cause differing outputs when differentiating or integrating...

