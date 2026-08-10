import numpy as np
import matplotlib.pyplot as plt

C = 299792458.0
FREQ = 10.7e9
D_ST = 0.863
lam = C / FREQ
D_LAMBDA_TX = D_ST / lam

# ITU-R S.1528 Tx Gain Function
def itu_s1528_tx(psi, D):
    """
    Computes ITU-R S.1528 Transmit Antenna Gain (dBi) given:
    psi: off-axis angle in degrees
    D: normalized antenna diameter (D/lambda)
    """
    psi = max(psi, 1e-9)
    G = 20 * np.log10(D) + 7.7
    pb = np.sqrt(1200) / D
    Y = 1.5 * pb
    Z = Y * 10**(0.04 * (G - 6.75))
    
    if psi <= Y:
        return G - 3 * (psi / pb)**2
    if psi <= Z:
        return G - 6.75 - 25 * np.log10(psi / Y)
    return 0

itu_s1528_tx_vec = np.vectorize(itu_s1528_tx)

off_axis_angles = np.linspace(0, 180, 1000)
tx_gains = itu_s1528_tx_vec(off_axis_angles, D_LAMBDA_TX)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(off_axis_angles, tx_gains, color='#007acc', linewidth=2, label=f'Tx Pattern (D/λ ≈ {D_LAMBDA_TX:.1f})')

plt.title("ITU-R S.1528 Transmit Antenna Gain Pattern", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Off-Axis Angle $\psi$ (Degrees)", fontsize=12)
plt.ylabel("Tx Antenna Gain (dBi)", fontsize=12)
plt.xlim(0, 180)
plt.ylim(bottom=-5)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()
