#!/usr/bin/env bash

holdings=(1 2 3 5 10 20 40 60)

for day in "${holdings[@]}"; do
    echo "Running portfolio optimization for holding period: ${day} days..."
    uv run python -m tyche.portfolio.run --holding "$day"
done
