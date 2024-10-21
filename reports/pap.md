---
title: Analysis of Patient Movement Behaviour for controlling Carbapenamase-producing Enterobacterialae in the Victorian healthcare system
author:
    - David Wu
    - Tjibbe Donker
    - etc
    - Andrew Stewardson
    - Michael Lydeamore
---

# Intro 

CPE is bad
It is in VIC, though it is currently contained
It is likely imported.
We want to understand how it would spread if introduced elsewhere.

Our minor objective here is to see if the pattern of patient movement within the system is "stable" in some sense, so that we drop the usage of the explicitly temporal network.

# Results

We find that

# Materials and Methods

We have data from Centre for Victorian Data Linkage that is line-listed in-patient presence data in hospitals and other healthcare facilities.
These include ...

We consider "transfers" as described in HospitalNetwork package...

These transfers have both temporal and spatial information.

We aggregate this information by choosing a temporal "window". All movements within the window are summed. 

This constructs a temporally aggregated network of nodes that can be visually conceptualised as:

We note that we have transfers that involve two nodes that are different in both location and time. This differs to the usual temporal network representation used in, for example, networking applications, where the duration of the edge is short compared to the timeframe of the temporal network. 
We thus also construct an intermediate representation, where a static network is generated at each time point, akin to a snapshot representation of a temporal network.
This snapshot representation acts like a static network for a given duration.

We perform direct network analysis on the temporal network and its snapshots.

First we consider the time series of certain network characteristics of the snapshots.
- centrality
- etc

These suggest qualitatively that there is little structural change in the system, bar some anomalies with openings and closings of different institutions.

Secondly we perform direct stochastic simulation of an SIS model, which assumes constant population size at each hospital. One statistic we don't have is the "size" of a hospital, so we estimate this based on the number of transfers out of the hospital (including self-transfers), and the proportion of all patients that return to the healthcare system.

This direct stochastic simulation is performed for 3 different types of networks, with subtly differing models.
[describe the models]

# Discussion

The snapshot representation defects.