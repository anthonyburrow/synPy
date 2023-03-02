import numpy as np
from astropy.io import fits 


try:
    extinc_found = True
    import extinction
except ModuleNotFoundError:
    extinc_found = False


def read_sp_data_fits(filename):
    '''.fits reader from Eddie'''
    hdulist = fits.open(filename)
    cards = hdulist[ 0 ].header

    # Check if using IRAF style with 4 axes. Assume flux is in AXIS4
    if "CDELT1" in cards: # FITS Header
        wldat = cards[ "CRVAL1" ] + cards[ "CDELT1" ] * np.arange( cards[ "NAXIS1" ] )
        fldat = hdulist[ 0 ].data  
    elif "CD1_1" in cards: # IRAF Header
        wldat = cards[ "CRVAL1" ] + cards[ "CD1_1" ] * np.arange( cards[ "NAXIS1" ] )
        fldat = hdulist[ 0 ].data[0][:]
        help_ = fldat.shape
        if len(help_) != 1:
            fldat = fldat[0,:]
    else:
        raise ValueError("No wl scale")

    hdulist.close()
    index = fldat.ravel().nonzero()
    wldat = wldat[index]
    fldat = fldat[index]

    return np.c_[wldat, fldat]


def read_data(fn, z=None, wave_range=None, normalize=True, ebmv=None, rv=3.1):
    if fn[-4:] == 'fits':
        data = read_sp_data_fits(fn)
    else:
        data = np.loadtxt(fn)

    # De-reshift and prune based on wavelength range
    if z is not None:
        data[:, 0] /= 1. + z

    # Prune to wavelength range
    if wave_range is not None:
        mask = (wave_range[0] <= data[:, 0]) & (data[:, 0] <= wave_range[1])
        data = data[mask]

    # Correct for extinction
    if ebmv is not None:
        if extinc_found:
            extinc = extinction.fitzpatrick99(data[:, 0], -rv * ebmv, rv)
            data[:, 1] = extinction.apply(extinc, data[:, 1])
        else:
            print(f'extinction module not found; this will not apply ebmv = {ebmv}')

    # Normalize
    if normalize:
        max_flux = data[:, 1].max()
        data[:, 1] /= max_flux

    try:
        data[:, 2] /= max_flux
    except IndexError:
        pass

    return data
