#!/usr/bin/env bash

metrics=("pagerank" "harmonic" "closeness" "betweenness" "eigenvector")

for metric in ${metrics[@]}; do
    python projection_metrics.py plot $1 -g -m $metric
done