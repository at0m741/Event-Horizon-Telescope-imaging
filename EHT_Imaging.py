import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

file_path = 'ER6_SGRA_2017_097_hi_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits'
hdulist = fits.open(file_path)

data = hdulist[0].data
header = hdulist[0].header

param_names = [header[f'PTYPE{i+1}'] for i in range(header['PCOUNT'])]
param_scales = [header[f'PSCAL{i+1}'] for i in range(header['PCOUNT'])]
print(param_names)
u_index = param_names.index('UU---SIN')
v_index = param_names.index('VV---SIN')
u = data.par(u_index) * param_scales[u_index]
v = data.par(v_index) * param_scales[v_index]

real = data['DATA'][:, 0, 0, 0, 0, 0, 0]
imag = data['DATA'][:, 0, 0, 0, 0, 0, 1]
amplitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)

grid_size = 2048
uv_grid = np.zeros((grid_size, grid_size), dtype=complex)
u_scaled = ((u - u.min()) / (u.max() - u.min()) * (grid_size - 1)).astype(int)
v_scaled = ((v - v.min()) / (v.max() - v.min()) * (grid_size - 1)).astype(int)

for i in range(len(u)):
    uv_grid[u_scaled[i], v_scaled[i]] += real[i] + 1j * imag[i]
    print(uv_grid[u_scaled[i], v_scaled[i]])

image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))
image_abs = np.abs(image)

fig, ax = plt.subplots(figsize=(10, 10))
img = ax.imshow(image_abs / np.max(image_abs), cmap='inferno', origin='upper', extent=[-1, 1, -1, 1])
ax.scatter(u / np.max(np.abs(u)), v / np.max(np.abs(v)), s=1, alpha=0.5, color='cyan', label='UV Coverage')
ax.scatter(-u / np.max(np.abs(u)), -v / np.max(np.abs(v)), s=1, alpha=0.5, color='magenta', label='Conjugate UV Coverage')
ax.set_xlabel('X (arcseconds)')
ax.set_ylabel('Y (arcseconds)')
ax.set_title('Reconstructed Image with UV Coverage')
ax.legend()

plt.colorbar(img, ax=ax, label='Normalized Intensity')
plt.show()

bins = np.linspace(0, np.max(np.sqrt(u**2 + v**2)), 50)
binned_amplitude, _ = np.histogram(np.sqrt(u**2 + v**2), bins=bins, weights=amplitude)
counts, _ = np.histogram(np.sqrt(u**2 + v**2), bins=bins)
binned_amplitude /= counts
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, binned_amplitude, '-', alpha=0.8, label='Binned Average Amplitude')
plt.scatter(np.sqrt(u**2 + v**2), amplitude, s=2, alpha=0.3, label='Individual Amplitudes', color='red')
plt.xlabel('UV Distance (wavelengths)')
plt.ylabel('Amplitude')
plt.title('Amplitude of Visibilities')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.sqrt(u**2 + v**2), phase, '.', alpha=0.5, label='Phase', color='blue')
plt.scatter(np.sqrt(u**2 + v**2), phase, s=2, alpha=0.3, label='Individual Phases', color='red')
plt.xlabel('UV Distance (wavelengths)')
plt.ylabel('Phase (radians)')
plt.title('Phase of Visibilities')
plt.legend()
plt.grid()
plt.show()
