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

12 Feb 2025
===========

We had a brief look at the behaviour of the power law fit as time progresses.
It looks like it's mostly stable, but the intercept (or mult. constant) increases over time, and the power seems to drop slightly over time.
There's also pretty obvious seasonal effects (of order week and year).

13 Feb 2025
===========

Today we'll look at the spectra of the Q matrix of the static network to see if there is going to be a problem with its spread.

<!-- ![](hitting_time_analysis/eigenvalues_Q_minus1.png) -->

We can see that we span eigenvalues from 1e-1 to 1e-5, which is probably going to be a problem. We also can't eliminate any of these since there isn't a large dropoff in the spectra until we get to the final few eigenvalues.

19 Feb 2025
===========

We can empirically solve y' = Qy and then linearly regress (with a log transform on y) to get approximate exponential decays.
This gives a range of $\lambda \sim [0.0005 - 1e5]$

This means that a scaling factor of roughly 1e3 will actually capture the integral correctly.


24 Feb 2025
===========

So we have, in our graph a node with an exit count of 10.
The timespan is 28.
The capacity of that node is 30.

So we get (1/28 * 1/3) of the target number moving per time unit. This is about 1%.

```{python}
In []: simulation.rng.multinomial(np.ones((5,), dtype=int)*100_000, simulation.parameters['transition_matrix'])
Out[]: 
array([[    0, 10581,     0,  1715, 87704],
       [    0,     0,  1490,  2984, 95526],
       [ 2718,     0,     0,     0, 97282],
       [    0,     0,  4437,     0, 95563],
       [    0,     0,     0,  1181, 98819]])
```

The problem is in the underlying transition matrix, which isn't being correctly normalised in the naive static case.
We should, in theory, check all other simulations to see if they also exhibit this behaviour

25 Feb 2025
===========

Computing the correct probability of return.
This is for the static return model (probably use in the snapshot model too)
We currently do

return ~ Binomial( T * dt )

where T is the observed rate of return per individual, and dt is the time step.
This is slightly problematic, since it restricts dt in a sense. If T * dt > 1.0, then we have a problem.

What we really want is that 

t ~ Expon(T)

p ~ P(t < dt)

then draw

return ~ Binomial( p )

We thus need the cdf of t: $1 - \exp(-T * t)$
and so $p = 1 - e^{-T * dt}$

The amount of deviation, especially at small T is relatively negligible.
A Taylor expansion about $t=0$ explains this:

$dp/dt = T e ^ {-T * t} \implies p(t) \approx 0 + T*1 (t - 0) + \mathcal{O}(t^2) = T * t$


---
We are also going to perform small case on a home+static network. We are currently blocked by the hardcoding of some attributes.
Like with naive static, we should either implement a function that accepts a given Graph and attr keys, or maybe even just accept some transition matrix components.

27 Feb 2025
===========

We discovered last night that there is a nice-ish relationship between the variance of an RV and the survival function.

$$Var[X] = 2 \int_0^\infty u S(u) du - \left[\int_0^\infty S(u) du\right] ^2$$

I did a brief experiment last night on a v. small case, and it seems like the variance is quite large.

We are also more interested in the distribution of the collection of RVs $\{X_{ij}\}$, i.e. the hitting time distribution. I guess that we cant really say much about this.
I guess we could say that we are (uniformly) drawing a random item from the collection, so we might be able to get something there? 
[gippity] Law of Total Variance 
Var(X_n) = Mean of Variances + Variance of Means
This is relatively computable.

This is important when comparing our hitting time eCDF to the distribution of expectations of hitting times, since we aren;'t actually comparing the same random variables... (the eCDF is prob. closer to a Gaussian with mean and variance based on the above (samples from X_i)) -> actually, this is not the case, since the eCDF still kinda looks Exponential.

4 March 2025
============
Wanting to make sure that the link times on the small case are representative of the "true" network.

We should examine by seed node, how the various hitting time dists behave relative to each other.

We could do this by checking different ts, or by checking the median hitting t.

We should also do this per target node.


6 Mar 2025
==========

Run the snapshot model with the static model as the snapshots (many times).
I would have to change the rate parameter slightly.

13 Mar 2025
===========

Looking at the movements, the static reaches equilibrium much faster than the snapshot.
We should also check the temporal network

[14/03] Checking the connectedness of the snapshot graphs -> the naive igraph union of the snapshots are not strongly connected, but the static network is...
  - turns out it's the isolated node that is causing this (and why we are 337 vs 338)
    - we can remove self-loops, purge isolated nodes and do this connectedness check over time.

[17/03]
This was done, and it looks like the connectedness of the aggregated graph roughly matches with the movement.

18 Mar 2025
===========

Compare the static and snapshot pairwise by node, to see if static is "always" slower than snapshot

Rare path - how to investigate?
  - Make a simple case triangle network + 1, where the +1 is kjoined with low weight in one (poss early) snapshot
  - Make table of t | edge | weight | rel weight | ... to detect rare edges in our network

25 Mar 2025
===========

Time to write up

[26/03]

Clean upnthe repository. Shelve stuff in the right place.
A fresh template has beens tarted in echidna_cleaup

[27/03]

Repo restructured we now have a "library" folder that is installable for core behaviour (graph projection, simulation, analysis)
various other folders for scripts and notebooks and experiments
and a potentially unified output folder - though we can and should reconstruct outputs in quarto in computation cells.

For release, we should write tests for the ocre library code, and we need to write up the entry points in the pyproject.toml

[3/04]

Add refs in todobib


[10/04]

Look at (rhs) outliers of the deviation [burstiness] and see if they are classical bursty
esp > 1.0
and larrgeish and smallish

- [01/05] so it looks like it's just one node with a limited time span of existence, causing a large rate (since denom [est size] is small). (8386) 
  - this prob means we need to revisit the size estimation, and divide by the existence span, instead of the entire time period

[15/05] 
- looking at our assumption of "churn rate" for size estimation. We have this on CVDL, but mean of mean etc gives 2d, 4d, 6hr. I want to extract this as
  - FID | Admissions Count | Mean Stay | Med Stay | Std Dev Stay | ...
  - and scramble by FID. I would expect _a_ summary stat to come out near 1
- Also did some figure splicing of transfers over time, which turned out nice, need to adjust some thigs for clarity, otherwise OK.
- We need to think of at least 1 more figure to include as a primary result.


[20/05]
- look at prop. at home
  - exists as a script, now need to clean up the plot
    - label overlaps, time concordance


[03/06]
- this package: https://ctan.org/pkg/tikz-network
  - can recreate the manim diagrma we drew, looks nice, haven't seen compat

[06/06]
- intro needs some more background research: we need to find non-tjibbe papers
  - Bruce Lee
  - Piotrowska
  - (france?)
    -


[12/06] ECHIDNA MEETING

- why not also look at farms (animal outbreak studies)
  - in the context of hospital-level R0 for static

[23/06]

Thinking a bit about the applicability of these results to "epidemics".
We are getting results about the Q matrix of the movement matrix.
We can augment this in a system where:
- S' = -bSI - QS + aI
- I' = bSI - QI - aI

where bSI and aI are diagonal. 

Some linear algebra later, we can maybe get some info, but we need that <a> is const over all locs
And I'm not sure if we can get any info about the block...

using FV, NGM:

F = ..?
V = ..?

- need to check in office with Math Epi ref.

[24/06]

- Looks like there is another group, loosely assoc with Piotrowska, but perhaps more senior:
  - Vitaly Belik
  - Rafael Mikolajczyk

Need to go into CVDL and extract the histogram of stay duration as a csv, then ask for export.

[01/07]

- ML has suggested moving the actual data charaacterisation bit to the results, and just presenting the assumptions in previous lit in the mats and methods. This seems like a good idea.

[8/07]

- Need to think of proper and good ways of generating a goodness of fit test for the PMFs that we have (binned histograms). prob a chi2, 
- I want to think about how to best do this test per-facility once we have pulled los from the VM. I think we can't so prob have to do inside VM, which reqquries a groupby regression. -- 74895640 -- SO provides a solution.
- mvoed some things around in pap, feels better, need to clean methods section, maybethink about how to incorp fig sproperly.
- I want to make a comment about the simulation time. I thikn this is present int he sim ids of the temporal and snapshot, but they are in different formats.
  - looked into, and it looks like we only recorded up to seconds detail, we needed more detail. temporal is 1-2s, snapshiot is 2-4 s.


[18/07]

- catastrophic acnecllation for chi2 test when CDF gets very small in expon.


[22/07]

- doe the los being pow-law dist mean that we dont get poiss dist depatures from a fcaility.