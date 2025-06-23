# quantum_visualize.py (fixed version with debugging)

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast

# === Load Data ===
df = pd.read_csv("quantum_data_export.csv")

# Debug: Show first few entries
print("\n🔍 Sample entries:")
print(df[['text', 'label', 'top_counts']].head(3))

# Convert safely
def safe_parse(d):
    try:
        return ast.literal_eval(d)
    except:
        return {}

df['top_counts'] = df['top_counts'].apply(safe_parse)

# Debug: Check if any rows are empty
empty_rows = df[df['top_counts'].apply(lambda x: not bool(x))]
print(f"\n⚠️ Empty rows found: {len(empty_rows)}")

# === Separate
ham = df[df['label'] == 0]
spam = df[df['label'] == 1]

def aggregate_bitstrings(subset):
    agg = Counter()
    for row in subset['top_counts']:
        agg.update(row)
    return dict(agg.most_common(10))

ham_top = aggregate_bitstrings(ham)
spam_top = aggregate_bitstrings(spam)

# Debug print
print("\n📊 Top Ham Bitstrings:", ham_top)
print("📊 Top Spam Bitstrings:", spam_top)

# === Plot if available
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if ham_top:
    axes[0].bar(ham_top.keys(), ham_top.values(), color='skyblue')
    axes[0].set_title("Top Bitstrings - HAM")
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, "No HAM data", ha='center')

if spam_top:
    axes[1].bar(spam_top.keys(), spam_top.values(), color='tomato')
    axes[1].set_title("Top Bitstrings - SPAM")
    axes[1].tick_params(axis='x', rotation=45)
else:
    axes[1].text(0.5, 0.5, "No SPAM data", ha='center')

plt.tight_layout()
plt.show()
