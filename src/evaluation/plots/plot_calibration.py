import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
# Define the output directory where your calibration npz files are saved.
output_dir = "/remote/gpu01a/pietschke/EoRFlow/output/paper_plots"

# Define the file paths for the calibration data.
# These files should have been saved earlier, for example:
# np.savez(os.path.join(output_dir, 'log_prob_ranks_pure.npz'), overall_ranks=all_ranks_pure)
# np.savez(os.path.join(output_dir, 'log_prob_ranks_noise.npz'), overall_ranks=all_ranks_noise)
pure_file = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-6_bigNoise/evaluation_filter/log_prob_ranks.npz'
noise_file = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit/evaluation/log_prob_ranks.npz'

# Load the rank data.
data_pure = np.load(pure_file)
ranks_pure = data_pure["overall_ranks"]   # adjust key if necessary

data_noise = np.load(noise_file)
ranks_noise = data_noise["overall_ranks"]

# Define bins for the credibility levels.
bins = np.linspace(0, 1, 50)

# Compute the quantiles for the empirical coverage.
# Here we use 1 - ranks (assuming lower rank means higher credibility).
quantiles_pure = np.quantile(1.0 - ranks_pure, bins)
quantiles_noise = np.quantile(1.0 - ranks_noise, bins)

# Start plotting.
plt.figure(figsize=(8, 8))

plt.plot(quantiles_pure, bins, 'b-', linewidth=2, label='Noiseless')
plt.plot(quantiles_noise, bins, 'r-', linewidth=2, label='Mock')
plt.plot([0, 1], [0, 1], 'black', alpha=0.8, linewidth=2, label='Ideal Coverage')

plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)
plt.xlabel('Credibility Level', fontsize=30)
plt.ylabel('Empirical Coverage', fontsize=30)
#plt.title('Log Probability Coverage Calibration', fontsize=16)
#plt.grid(True, alpha=0.3)
plt.legend(fontsize=30)
plt.tight_layout()

# Save the combined calibration plot.
out_path = os.path.join(output_dir, "log_prob_coverage_combined.pdf")
plt.savefig(out_path, dpi=400, bbox_inches='tight')
plt.show()