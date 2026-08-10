import numpy as np
import matplotlib.pyplot as plt

C = 299792458.0
FREQ = 10.7e9           # 10.7 GHz Ku-band
D_UT = 0.6              # User Terminal diameter (0.6 meters)

lam = C / FREQ          # Wavelength ≈ 0.028 m
D_LAMBDA_RX = D_UT / lam  # D/lambda ≈ 21.41

# ITU-R S.1428 Rx Gain Function
def itu_s1428_rx(phi_arr, D_lambda):
    """
    Computes ITU-R S.1428 Receive Antenna Gain (dBi) given:
    - phi_arr: off-axis angle in degrees (array or scalar)
    - D_lambda: normalized antenna diameter (D/lambda)
    """
    phi = np.maximum(phi_arr, 1e-9)
    
    D = np.clip(D_lambda, 20.0, 25.0)
    
    Gmax = 20 * np.log10(D) + 7.7
    G1 = 29 - 25 * np.log10(95.0 / D)
    phim = np.sqrt((Gmax - G1) / 2.5e-3) / D
    phir = 95.0 / D
    
    cond1 = phi < phim
    cond2 = (phi >= phim) & (phi < phir)
    cond3 = (phi >= phir) & (phi < 33.1)
    cond4 = (phi >= 33.1) & (phi < 80.0)
    cond5 = phi >= 80.0
    
    g1 = Gmax - 2.5e-3 * (D * phi)**2
    g2 = np.full_like(phi, G1)
    g3 = 29.0 - 25.0 * np.log10(phi)
    g4 = np.full_like(phi, -9.0)
    g5 = np.full_like(phi, -5.0)
    
    return np.select([cond1, cond2, cond3, cond4, cond5], [g1, g2, g3, g4, g5])

off_axis_angles = np.linspace(0, 180, 1000)
rx_gains = itu_s1428_rx(off_axis_angles, D_LAMBDA_RX)

plt.figure(figsize=(10, 5), dpi=100)
plt.plot(
    off_axis_angles, 
    rx_gains, 
    color='#e74c3c', 
    linewidth=2, 
    label=f'Rx Pattern ($D_\\text{{UT}} = 0.6\\text{{m}}$, $D/\\lambda \\approx {D_LAMBDA_RX:.1f}$)'
)

plt.title("ITU-R S.1428 Receiver Antenna Gain Pattern", fontsize=13, fontweight='bold', pad=12)
plt.xlabel(r"Off-Axis Angle $\phi$ (Degrees)", fontsize=11)
plt.ylabel("Rx Antenna Gain (dBi)", fontsize=11)
plt.xlim(0, 180)
plt.ylim(bottom=-15, top=max(rx_gains) + 3)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()
