import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# Read the CSV files
csv_path = 'Training results'
dncnn_b_df = pd.read_csv(os.path.join(csv_path, 'DnCNN-B-50epochs.csv'))
dncnn_s_df = pd.read_csv(os.path.join(csv_path, 'DnCNN-S-50epochs.csv'))

# Create figure with 2 side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Loss through epochs (LEFT)
ax1.plot(dncnn_b_df['epoch'], dncnn_b_df['loss'], 'b-', linewidth=2, marker='o', markersize=4, label='DnCNN-B')
ax1.plot(dncnn_s_df['epoch'], dncnn_s_df['loss'], 'r-', linewidth=2, marker='s', markersize=4, label='DnCNN-S')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('MSE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, max(max(dncnn_b_df['epoch']), max(dncnn_s_df['epoch'])))

# Plot 2: PSNR Out through epochs (RIGHT)
ax2.plot(dncnn_b_df['epoch'], dncnn_b_df['psnr_out'], 'b-', linewidth=2, marker='o', markersize=4, label='DnCNN-B')
ax2.plot(dncnn_s_df['epoch'], dncnn_s_df['psnr_out'], 'r-', linewidth=2, marker='s', markersize=4, label='DnCNN-S')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('PSNR Out (dB)')
ax2.set_title('PSNR')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, max(max(dncnn_b_df['epoch']), max(dncnn_s_df['epoch'])))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('training_results_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print some statistics
print("Training Results Summary:")
print("=" * 50)
print(f"DnCNN-B Final PSNR Out: {dncnn_b_df['psnr_out'].iloc[-1]:.4f} dB")
print(f"DnCNN-S Final PSNR Out: {dncnn_s_df['psnr_out'].iloc[-1]:.4f} dB")
print(f"DnCNN-B Final Loss: {dncnn_b_df['loss'].iloc[-1]:.6f}")
print(f"DnCNN-S Final Loss: {dncnn_s_df['loss'].iloc[-1]:.6f}")
print(f"DnCNN-B Max PSNR Out: {dncnn_b_df['psnr_out'].max():.4f} dB")
print(f"DnCNN-S Max PSNR Out: {dncnn_s_df['psnr_out'].max():.4f} dB") 