import numpy as np
import matplotlib.pyplot as plt
import time

# Given data
median_mean_g = [
    2.55, 2.65, 2.75, 2.85, 2.95, 3.05, 3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.85, 3.95, 4.05, 4.15, 4.25,
    4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05,
    6.15, 6.25, 6.35, 6.45, 6.55, 6.65, 6.75, 6.85, 6.95, 7.05, 7.15, 7.25, 7.35, 7.45, 7.55, 7.65, 7.75, 7.85,
    7.95, 8.05, 8.15, 8.25, 8.35, 8.45, 8.55, 8.65, 8.75, 8.85, 8.95, 9.05, 9.15, 9.25, 9.35, 9.45, 9.55, 9.65,
    9.75, 9.85, 9.95, 10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75, 10.85, 10.95, 11.05, 11.15, 11.25,
    11.35, 11.45, 11.55, 11.65, 11.75, 11.85, 11.95, 12.05, 12.15, 12.25, 12.35, 12.45, 12.55, 12.65, 12.75, 12.85,
    12.95, 13.05, 13.15, 13.25, 13.35, 13.45, 13.55, 13.65, 13.75, 13.85, 13.95, 14.05, 14.15, 14.25, 14.35, 14.45,
    14.55, 14.65, 14.75, 14.85, 14.95, 15.05, 15.15, 15.25, 15.35, 15.45, 15.55, 15.65, 15.75, 15.85, 15.95, 16.05,
    16.15, 16.25, 16.35, 16.45, 16.55, 16.65, 16.75, 16.85, 16.95, 17.05, 17.15, 17.25, 17.35, 17.45, 17.55, 17.65,
    17.75, 17.85, 17.95, 18.05, 18.15, 18.25, 18.35, 18.45, 18.55, 18.65, 18.75, 18.85, 18.95, 19.05, 19.15, 19.25,
    19.35, 19.45, 19.55, 19.65, 19.75, 19.85, 19.95, 20.05, 20.15, 20.25, 20.35, 20.45, 20.55, 20.65, 20.75, 20.85,
    20.95, 21.05, 21.15, 21.25, 21.35, 21.45, 21.55, 21.65, 21.75, 21.85, 21.95, 22.05, 22.15, 22.25, 22.35, 22.45,
    22.55
]
number_of_stars = [
    10, 11, 13, 15, 17, 20, 23, 27, 31, 35, 40, 46, 53, 61, 70, 81, 91, 100, 112, 125, 141, 158, 177, 198, 223, 250,
    279, 312, 349, 391, 437, 489, 547, 612, 684, 765, 856, 957, 1070, 1197, 1338, 1496, 1673, 1871, 2092, 2340,
    2595, 2878, 3192, 3540, 3926, 4354, 4829, 5355, 5939, 6586, 7227, 7922, 8676, 9494, 10382, 11347, 12397, 13540,
    14785, 16140, 17483, 18933, 20500, 22194, 24027, 25995, 28117, 30000, 31798, 33703, 35721, 37861, 40130, 42537,
    45091, 47802, 50681, 53738, 56985, 60434, 64098, 68000, 72000, 76400, 81000, 85900, 91200, 96800, 102800,
    109200, 116000, 123000, 130800, 139000, 147800, 157200, 167200, 177800, 189100, 201200, 214100, 227800, 242500,
    258000, 274600, 292100, 310800, 330700, 351800, 374200, 398100, 423500, 450500, 479300, 509900, 542500, 577200,
    614200, 653600, 695500, 740100, 787500, 838000, 891700, 948800, 1009000, 1074000, 1143000, 1216000, 1294000,
    1377000, 1465000, 1559000, 1659000, 1765000, 1878000, 2000000, 2129000, 2267000, 2415000, 2573000, 2742000,
    2924000, 3118000, 3327000, 3550000, 3788000, 4044000, 4318000, 4611000, 4925000, 5262000, 5623000, 6012000,
    6428000, 6875000, 7354000, 7868000, 8420000, 9012000, 9648000, 10330000, 11060000, 11840000, 12680000, 13580000,
    14550000, 15580000, 16680000, 17860000, 19120000, 20470000, 21910000, 23450000, 25100000, 26860000, 28750000,
    29367000, 29990000, 27650000, 23780000, 17210000, 12450000, 9000000, 6500000, 4700000, 3400000, 1930000,
    1090000, 620000, 350000, 200000, 116000, 67000, 39000
]

# Convert to numpy arrays
bin_centers = np.array(median_mean_g) - 0.43
counts_full_sky = np.array(number_of_stars)

# Calculate full-sky area in square arcmin
full_sky_sq_deg = 4 * 180 ** 2 / np.pi
full_sky_sq_arcmin = full_sky_sq_deg * 3600
print(full_sky_sq_arcmin)
# Compute density per square arcmin
density_per_bin = counts_full_sky / full_sky_sq_arcmin
n_bins = len(bin_centers)

plt.plot(bin_centers, density_per_bin)
plt.yscale('log')
plt.ylabel(r'$star/arcmin^2$')
plt.xlabel(r'Tmag')
plt.show()

# Constants
FWHM = 0.5  # arcmin
sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
print(f"PSF sigma: {sigma:.4f} arcmin")

# Precompute base fluxes for each bin
base_fluxes = 10 ** (-0.4 * bin_centers)

# Simulation parameters
N_sim = 100000  # Number of simulations
print(f"Starting {N_sim} simulations...")

# Array to store flux per bin per simulation
flux_per_bin = np.zeros((n_bins, N_sim))

start_time = time.time()

# Run simulations
for sim_idx in range(N_sim):
    for bin_idx in range(n_bins):
        # Draw number of stars from Poisson distribution
        k = np.random.poisson(density_per_bin[bin_idx])

        if k > 0:
            # Generate random positions
            x = np.random.uniform(0, 1, size=k)
            y = np.random.uniform(0, 1, size=k)

            # Calculate distances from center (0.5, 0.5)
            dx = x - 0.5
            dy = y - 0.5
            r2 = dx ** 2 + dy ** 2

            # Calculate PSF weights
            g = np.exp(-r2 / (2 * sigma ** 2))

            # Sum PSF-weighted fluxes for this bin
            flux_per_bin[bin_idx, sim_idx] = base_fluxes[bin_idx] * np.sum(g)

    # Progress tracking
    if (sim_idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Completed {sim_idx + 1}/{N_sim} simulations. Time: {elapsed:.2f} sec")

# Calculate average flux per bin with uncertainties
mean_flux_per_bin = np.mean(flux_per_bin, axis=1)
std_flux_per_bin = np.std(flux_per_bin, axis=1)

# Target magnitudes (4 to 22 in steps of 0.5)
target_mags = np.arange(4.0, 22.1, 0.5)
n_targets = len(target_mags)

# Arrays for results
total_contam = np.zeros(n_targets)
brighter_contam = np.zeros(n_targets)
dimmer_contam = np.zeros(n_targets)

# Calculate contamination for each target magnitude
for i, mag in enumerate(target_mags):
    # Find index where bins transition from brighter to dimmer
    split_idx = np.searchsorted(bin_centers, mag)

    # Calculate contaminant fluxes
    brighter_flux = np.sum(mean_flux_per_bin[:split_idx])
    dimmer_flux = np.sum(mean_flux_per_bin[split_idx:])
    total_flux = brighter_flux + dimmer_flux

    # Target star flux
    target_flux = 10 ** (-0.4 * mag)

    # Calculate contamination ratios
    total_contam[i] = total_flux / target_flux
    brighter_contam[i] = brighter_flux / target_flux
    dimmer_contam[i] = dimmer_flux / target_flux

# Convert to percentages
total_contam_pct = total_contam * 100
brighter_contam_pct = brighter_contam * 100
dimmer_contam_pct = dimmer_contam * 100

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(target_mags, total_contam_pct, 'k-', lw=2, label='Total Contamination')
plt.plot(target_mags, brighter_contam_pct, 'b--', label='Brighter Stars')
plt.plot(target_mags, dimmer_contam_pct, 'r-.', label='Dimmer Stars')

# Format plot
plt.xlabel('Target Magnitude (Gaia G)')
plt.ylabel('Contamination Flux / Target Flux (%)')
plt.title('TESS Contamination: Flux Ratio to Target Star')
plt.yscale('log')
plt.ylim(1e-4, 1e4)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save and show
# plt.savefig('tess_contamination_flux_ratio_optimized.png', dpi=300)
plt.show()