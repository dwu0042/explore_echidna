---
title: Prototype Network Simulation
date: Approximately 7 December 2023
---

## Motivation

We want to see if simulations of CPE spread as modulated by patient movement is significantly different between a fully temporal network, a discretised snapshot-representation of a temporal network, and an aggregated static network.

This is to see if we can just use the aggregated static network (or a random temporal network snapshot) in lieu of a more expensive setup

## Method - Networks

We have line list data of patient admissions in the Victorian hospital system. We use these line lists to generate a fully temporal network by constructing an edge E between locations L_p and L_q at times t_i and t_j (resp.) if there is a patient that leaves (is discharged) from L_p at time t_i and then subsequently is admitted to L_q at time t_j. We note that t_i can equal t_j (direct transfer). This network cannot be used as it can contain identifiable data of patients, ...

We instead extract a series of partially discretised temporal networks, where the times t_i are rounded to the nearest finite time point...

Then, from each of these partially discretised temporal networks, we generate a timeseries of static networks, which capture a subset of the edges of the partially discretised temporal network that are incident on nodes of each discretised time step.

The aggregate static network is a single network, constructed by aggregating all edges out of hospitals...

## Method - Simulation

We use stochastic metapopulation models. This is an SIS model with additional movement. We assume one a single seeding event. We can write down analogous ODE models:

$$\begin{aligned}
    \frac{dS_p}{dt} &= -\beta \frac{S_p I_p}{S_p + I_p} + \gamma I_p \\
    \frac{dI_p}{dt} &= \beta \frac{S_p I_p}{S_p + I_p} - \gamma I_p - \sum_q M_{pq} I_p + \sum_q M_{qp} I_p
\end{aligned}$$

We only model movement on the infectives, since we assume that $S_p+I_p$ is roughly constant (a hospital is always roughly at capacity, and movements out to other hospitals or out of the system are replenished by movements from other hospitals and into the system).

In order to take largish steps, we model events in one timestep occurring independently from each other, i.e. new infections in the same timestep do not increase infection pressure. This is roughly OK for CPE spread.

This results in three-ish types of events:

| Type of Event  | Rate, $h$              | Number of Events Distribution            |
|-----------------|--------------------|-----------------------------------|
| Infection      | $\beta \frac{SI}{S+I}$ | $\text{Poisson}\left(h \Delta{t}\right)$ |
| Recovery       | $\gamma I$             | as above                                 |
| Movement (in)  |                        | as above                                 |
| Movement (out) |                        | as above                                 |

We will run into issues if $h\Delta{t}$ is too large (inaccuracy), or too small (incorrect probabilities)

## Preliminary Results

None

## Round 2 Model

We fully decouple S from the system by stating that N is fixed and $S + I = N$, not particularly complex, but solves our issues with running into 0s.

This gives us curves that don't cause the system to accumulate people over time :)