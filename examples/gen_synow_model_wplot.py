from synPy import SynowModel
import matplotlib.pyplot as plt


global_params = {
    'synow_lines_path'     : '$HOME/.bin/synow/lines/',
    'kurucz_linelist_path' : '$HOME/data/kurucz_lines/',
    'refdata_path'         : '$HOME/.bin/synow/src/',

    'spectrum_file'        : 'synthetic.dat',

    'vphot'                : 5000.0,
    'vmax'                 : 40000.0,
    'tbb'                  : 6000.0,

    'stspec'               : 3000.0,
    'ea'                   : 3200.0,
    'eb'                   : 9000.0,
}

# SynowModel takes in Spextractor read arguments for fitting
read_params = {
    'data': 'SN2023bvj_gr4_NOT_AL_20230221.fits',
    'z': 0.00512
}

# This instantiates the model for fitting to a spectrum you give it via read_params
model = SynowModel(**read_params)

# Otherwise the default global parameters are used
# (you don't need to have them all in global_params, just the ones you want to change)
model.set_params(**global_params)

# Add features to include in model (parameters you don't give go to default values).
# These labels (e.g. 'H I') can be anything you want and are only meant as a reference
# to the fitting below.
feature = { 'an' : 1, 'ai' : 0, 'tau1' : 20., 'vmine' : 5, 'vmaxe' : 40,
            've' : 3.0 }
model.add('H I', **feature)

feature = { 'an' : 6, 'ai' : 1, 'tau1' : 3, 'vmine' : 5, 'vmaxe' : 40,
            've' : 1.0 }
model.add('C II', **feature)

# Fits the parameters (feat_params) of the "feature" such that least squares
# is minimized within the wave_range compared to the observed spectrum
fit_params = {
    'wave_range': (6000., 6900.),
    'feature': 'H I',
    'feat_params': [
        'tau1',
        've'
    ],
    'feat_bounds': [
        (0., 20.),
        (0.5, 3.0)
    ]
}

# model.fit(**fit_params)

# At any time you can generate a script based on current/optimized parameters
model.write_script()

# Generate a current script, run it, save to spectrum_file, and return the results
data_synth = model.get_synth()


# Plot results
data_obs = model.data

max_obs = data_obs[:, 1].max()
max_synth = data_synth[:, 1].max()
data_obs[:, 1] *= max_synth / max_obs

fig, ax = plt.subplots()

ax.plot(data_synth[:, 0], data_synth[:, 1], label='SYNOW')
ax.plot(data_obs[:, 0], data_obs[:, 1], label='Observed')

ax.set_ylabel(r'$F_\lambda$ (arbitrary units)')
ax.set_xlabel(r'$\lambda$ ($\mathrm{\AA}$)')

ax.set_xlim(global_params['ea'], global_params['eb'])
ax.set_ylim(bottom=0.)

plt.show()
