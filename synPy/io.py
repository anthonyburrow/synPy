import numpy as np


def read_data(fn, z=None, wave_range=None, normalize=True):
    data = np.loadtxt(fn)

    # De-reshift and prune based on wavelength range
    if z is not None:
        data[:, 0] /= 1. + z

    # Prune to wavelength range
    if wave_range is not None:
        mask = (wave_range[0] <= data[:, 0]) & (data[:, 0] <= wave_range[1])
        data = data[mask]

    # Normalize
    if normalize:
        data[:, 1] /= data[:, 1].max()

    return data
