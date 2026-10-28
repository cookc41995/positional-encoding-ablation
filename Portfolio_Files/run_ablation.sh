#!/bin/bash
# Remove old results if they exist so we start fresh
rm -f ablation_results.csv

for pe in none sinusoidal learnable
do
    for seed in 42 1024 2048
    do
        python train.py --pe_type $pe --seed $seed
    done
done
echo "Ablation study complete! Results saved to ablation_results.csv"
