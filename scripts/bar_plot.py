# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
nll_file_human = '../nll_data/poem_Human_llama3-8b-instruct.txt'
nll_file_chatgpt = '../nll_data/poem_ChatGPT_poem_llama3-8b-instruct.txt'
nll_file_tulu2 = '../nll_data/poem_Tulu2-dpo-70B_poem_llama3-8b-instruct.txt'

# %%
# Load nll
def load_nll(nll_file):
    nlls = []
    with open(nll_file, 'r') as infile:
        for line in infile:
            # Split the line into numbers and convert them to floats
            nlls.extend([float(num) for num in line.split()])
    return nlls

nlls_human = load_nll(nll_file_human)
nlls_chatgpt = load_nll(nll_file_chatgpt)
nlls_tulu2 = load_nll(nll_file_tulu2)

# %%
# Calculate statistics
def calculate_stats(nlls):
    """Calculate mean and standard deviation of NLLs"""
    return np.mean(nlls), np.std(nlls)

# Calculate statistics for each source
mean_human, std_human = calculate_stats(nlls_human)
mean_chatgpt, std_chatgpt = calculate_stats(nlls_chatgpt)
mean_tulu2, std_tulu2 = calculate_stats(nlls_tulu2)

print("NLL Statistics:")
print(f"Human - Mean: {mean_human:.4f}, Std: {std_human:.4f}")
print(f"ChatGPT - Mean: {mean_chatgpt:.4f}, Std: {std_chatgpt:.4f}")
print(f"Tulu2 - Mean: {mean_tulu2:.4f}, Std: {std_tulu2:.4f}")

# %%
# Create boxplot comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for boxplot
data = [nlls_human, nlls_chatgpt, nlls_tulu2]
sources = ['Human', 'ChatGPT', 'Tulu2']
colors = ['blue', 'green', 'red']

# Create boxplot
box_plot = ax.boxplot(data, labels=sources, patch_artist=True, 
                      boxprops=dict(facecolor='lightgray', alpha=0.7),
                      medianprops=dict(color='black', linewidth=2),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=3))

# Color the boxes
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# ax.set_title('NLL Distribution Comparison (Boxplot)', fontsize=16, fontweight='bold')
ax.set_ylabel('NLL Value', fontsize=14)
# ax.set_xlabel('Data Source', fontsize=14)
ax.tick_params(axis='x', labelsize=14)  # Make x-axis tick labels larger
ax.grid(axis='y', alpha=0.3)

# Add statistics text for each box
stats_texts = [
    f'Mean: {mean_human:.3f}\nStd: {std_human:.3f}',
    f'Mean: {mean_chatgpt:.3f}\nStd: {std_chatgpt:.3f}',
    f'Mean: {mean_tulu2:.3f}\nStd: {std_tulu2:.3f}'
]

# Position text above each box
for i, (stats_text, color) in enumerate(zip(stats_texts, colors)):
    # Get the position of the box
    box_pos = i + 1
    # Get the maximum value of the box (whisker top)
    box_max = np.percentile(data[i], 75) + 1.5 * (np.percentile(data[i], 75) - np.percentile(data[i], 25))
    # Add text above the box
    ax.text(box_pos-0.3, box_max - 0.5, stats_text, ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig('nll_boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Additional statistical analysis
print("\nDetailed Statistical Analysis:")
print("=" * 50)

# Sample sizes
print(f"Sample sizes:")
print(f"Human: {len(nlls_human)} tokens")
print(f"ChatGPT: {len(nlls_chatgpt)} tokens")
print(f"Tulu2: {len(nlls_tulu2)} tokens")

# Min and max values
print(f"\nMin/Max values:")
print(f"Human: {np.min(nlls_human):.4f} / {np.max(nlls_human):.4f}")
print(f"ChatGPT: {np.min(nlls_chatgpt):.4f} / {np.max(nlls_chatgpt):.4f}")
print(f"Tulu2: {np.min(nlls_tulu2):.4f} / {np.max(nlls_tulu2):.4f}")

# Percentiles
print(f"\n25th, 50th, 75th percentiles:")
print(f"Human: {np.percentile(nlls_human, [25, 50, 75])}")
print(f"ChatGPT: {np.percentile(nlls_chatgpt, [25, 50, 75])}")
print(f"Tulu2: {np.percentile(nlls_tulu2, [25, 50, 75])}")

print("\nPlot saved as:")
print("- nll_boxplot_comparison.png")
