import numpy as np
import matplotlib.pyplot as plt

"""
This script calculates and visualizes the percentage loss in Shannon channel capacity as a function of the Interference-to-Noise Ratio (INR) across various baseline Signal-to-Noise Ratios (SNRs). It computes the theoretical Shannon capacity under clean conditions and compares it against capacity degraded by co-channel interference, generating a line plot to illustrate the throughput impact.
"""

inr_db = np.arange(-30, 31, 1)
inr_linear = 10 ** (inr_db / 10.0)

baseline_snrs_db = [3, 5, 10, 15, 20]

plt.clf()

for snr_db in baseline_snrs_db:
    snr_linear = 10 ** (snr_db / 10.0)
    
    c_clean = np.log2(1 + snr_linear)
    
    sinr_linear = snr_linear / (1 + inr_linear)
    c_dirty = np.log2(1 + sinr_linear)
    
    pct_drop = ((c_clean - c_dirty) / c_clean) * 100.0
    
    plt.plot(inr_db, pct_drop, label=f'Baseline SNR = {snr_db} dB', lw=2)

plt.title('Capacity Loss Percentage vs. Interference-to-Noise Ratio (INR)', fontsize=12, pad=15)
plt.xlabel('Interference-to-Noise Ratio (INR) [dB]', fontsize=10)
plt.ylabel('Capacity Loss (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(-30, 30)
plt.ylim(0, 100)
plt.xticks(np.arange(-30, 31, 10))
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
