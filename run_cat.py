import pandas as pd
import os
from tests.benchmark_all import (
    benchmark_categorical, CAT_DATA_DIR, subset_data, MAX_SAMPLES, OUTPUT_DIR, plot_results
)

print("Running categorical benchmarks...")
cat_results = {}
cat_files = [f for f in os.listdir(CAT_DATA_DIR) if f.endswith(".csv") and "subset" not in f]
for f in cat_files:
    filepath = os.path.join(CAT_DATA_DIR, f)
    subset_path = subset_data(filepath, MAX_SAMPLES)
    cat_results[f] = benchmark_categorical(subset_path)

print("Merging results...")
old_csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
if os.path.exists(old_csv_path):
    old_df = pd.read_csv(old_csv_path)
    sem_df = old_df[old_df['Category'] == 'SEM']
else:
    sem_df = pd.DataFrame()

flat_results = []
for ds, res in cat_results.items():
    for algo, metrics in res.items():
        flat_results.append({"Category": "Categorical", "Dataset": ds, "Algorithm": algo, **metrics})

cat_df = pd.DataFrame(flat_results)
new_df = pd.concat([cat_df, sem_df])
new_df.to_csv(old_csv_path, index=False)

plot_results(cat_results, "Categorical Algorithms Benchmark Time", os.path.join(OUTPUT_DIR, "categorical_benchmark.png"))
print("Done!")
