import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scienceplots
plt.style.use('science')
# Define the output directory where your calibration npz files are saved.
output_dir = "/remote/gpu01a/pietschke/EoRFlow/output/paper_plots"

# Define the file paths for the calibration data.
# These files should have been saved earlier, for example:
# np.savez(os.path.join(output_dir, 'log_prob_ranks_pure.npz'), overall_ranks=all_ranks_pure)
# np.savez(os.path.join(output_dir, 'log_prob_ranks_noise.npz'), overall_ranks=all_ranks_noise)
pure_file = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models/pure_10_512/evaluation_filter/log_prob_ranks.npz'
noise_file = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models/noise_10_512/evaluation/log_prob_ranks.npz'
aaStar_file = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d/aaStar_mod_ps2d_10_512/evaluation/log_prob_ranks.npz'
#aa4_file = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d/aa4_mod_ps2d_10_512/evaluation_gauss/log_prob_ranks.npz'

# Load the rank data.
data_aaStar = np.load(aaStar_file)
ranks_aaStar = data_aaStar["overall_ranks"]   

data_pure = np.load(pure_file)
ranks_pure = data_pure["overall_ranks"]

data_opt = np.load(noise_file)
ranks_opt = data_opt["overall_ranks"]

# Define bins for the credibility levels.
bins = np.linspace(0, 1, 50)

# Compute the quantiles for the empirical coverage.
# Here we use 1 - ranks (assuming lower rank means higher credibility).
quantiles_aaStar = np.quantile(1.0 - ranks_aaStar, bins)
quantiles_pure = np.quantile(1.0 - ranks_pure, bins)
quantiles_opt = np.quantile(1.0 - ranks_opt, bins)

# binomial confidence intervals for the empirical coverage
def binomial_ci(successes, trials, alpha=0.05):
    """Compute the binomial confidence interval using normal approximation."""
    if trials == 0:
        return 0.0, 0.0
    p_hat = successes / trials
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / trials)
    return max(0, p_hat - z * se), min(1, p_hat + z * se)

# Compute confidence intervals for the quantiles.
ci_aaStar = [binomial_ci(np.sum(1.0 - ranks_aaStar < q), len(ranks_aaStar)) for q in quantiles_aaStar]
ci_pure = [binomial_ci(np.sum(1.0 - ranks_pure < q), len(ranks_pure)) for q in quantiles_pure]
ci_opt = [binomial_ci(np.sum(1.0 - ranks_opt < q), len(ranks_opt)) for q in quantiles_opt]

# Start plotting.
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'black', alpha=0.8, linewidth=2, label='Ideal Coverage')
plt.plot(quantiles_pure, bins, 'slategray', linewidth=2, label='Noiseless')
plt.plot(quantiles_aaStar, bins, 'b-', linewidth=2, label='AA* mod')
plt.plot(quantiles_opt, bins, 'r-', linewidth=2, label='AA4 opt')

plt.fill_between(quantiles_pure, [c[0] for c in ci_pure], [c[1] for c in ci_pure], color='slategray', alpha=0.2)
plt.fill_between(quantiles_aaStar, [c[0] for c in ci_aaStar], [c[1] for c in ci_aaStar], color='blue', alpha=0.2)
plt.fill_between(quantiles_opt, [c[0] for c in ci_opt], [c[1] for c in ci_opt], color='red', alpha=0.2)

plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)
plt.xlabel('Credibility Level', fontsize=30)
plt.ylabel('Empirical Coverage', fontsize=30)
#plt.title('Log Probability Coverage Calibration', fontsize=16)
#plt.grid(True, alpha=0.3)
plt.legend(fontsize=30, loc='upper left')
plt.tight_layout()

# Save the combined calibration plot.
out_path = os.path.join(output_dir, "log_prob_coverage_final.pdf")
plt.savefig(out_path, dpi=400, bbox_inches='tight')
plt.show()