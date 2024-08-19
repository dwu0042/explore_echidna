# Underlying transmission model

## Structure

The state variable, $I$, is a vector of the number of infected individuals in each of $N$ hospitals.
There is also an auxiliary variable $D$, which is a vector or 2D tensor of the number of temporary departures in each of the $N$ hospitals ($\times$ time step of return, if 2D).

We index hospitals with $i$. We index time with $t$.

## Processes

0. Constant replenishment: $N_i = S_i + I_i$
1. Infection: $\lambda(I_i \to I_i+1) = \beta S_i I_i$
2. Recovery: $\lambda(I_i \to I_i-1) = \delta I_i$
3. Removal: $\lambda(I_i \to I_i-1) = \gamma I_i$
    1. probability of actual removal (never returning) = $p_i$
    2. probability of moving to another place := $1 - p_i$
       1. where to is governed by weight/departure/arrival
    3. $\gamma$ is tuned to the network weights, averaged over time? (based on the static network)

