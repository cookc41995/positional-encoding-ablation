import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pe_type', type=str, required=True, choices=['none', 'sinusoidal', 'learnable'])
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    # 1. Lock the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2. Setup Device and track VRAM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 3. Load Synthetic Data (Fall back to generating if missing)
    if not os.path.exists('demand_data.csv'):
        from generate_data import generate_synthetic_demand
        generate_synthetic_demand(num_days=30, seed=42)
    
    df = pd.read_csv('demand_data.csv')
    data = torch.tensor(df['demand_kw'].values, dtype=torch.float32).view(-1, 24) # Daily batches
    
    # 4. Define a Tiny Transformer-based architecture
    class TinyTimeSeriesTransformer(nn.Module):
        def __init__(self, pe_type):
            super().__init__()
            self.pe_type = pe_type
            self.linear = nn.Linear(24, 24)
            if pe_type == 'learnable':
                self.pe = nn.Parameter(torch.randn(1, 24))
            
        def forward(self, x):
            if self.pe_type == 'learnable':
                x = x + self.pe
            elif self.pe_type == 'sinusoidal':
                # Simplified sinusoidal addition
                seq_len = x.size(1)
                pe = torch.sin(torch.arange(seq_len, dtype=torch.float32) / 10000).unsqueeze(0).to(x.device)
                x = x + pe
            return self.linear(x)

    model = TinyTimeSeriesTransformer(args.pe_type).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 5. Training Loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, data.to(device))
        loss.backward()
        optimizer.step()

    # 6. Measure Inference Latency
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(data.to(device))
    latency_ms = (time.perf_counter() - start_time) * 1000

    # 7. Measure VRAM
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    # 8. Save results to CSV
    results_file = 'ablation_results.csv'
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['pe_type', 'seed', 'loss', 'latency_ms', 'peak_vram_gb'])
        writer.writerow([args.pe_type, args.seed, round(loss.item(), 4), round(latency_ms, 2), round(peak_vram_gb, 4)])

    print(f"Finished {args.pe_type} with seed {args.seed} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()