from synPy import SynowModel
from synPy.io import read_data


global_params = {
    'synow_lines_path'     : '$HOME/.bin/synow/lines/',
    'kurucz_linelist_path' : '$HOME/.bin/synow/kurucz_lines/',
    'refdata_path'         : '$HOME/.bin/synow/src/',

    'spectrum_file'        : 'synthetic.dat',

    'vphot'                : 10000.0,
    'vmax'                 : 40000.0,
    'tbb'                  : 17000.0,

    'stspec'               :  3000.0,
    'ea'                   :  3200.0,
    'eb'                   :  9000.0,
}

# For fitting
fn = '../spectra/my_spectrum.dat'
z = 0.014023


model = SynowModel()
model.set_params(**global_params)


# Manual Fitting and Model Setup

feature = { 'an' : 8, 'ai' : 0, 'tau1' : 0.5, 'vmine' : 12, 'vmaxe' : 40,
            've' : 1.0 }
model.add('O I', **feature)

feature = { 'an' : 12, 'ai' : 1, 'tau1' : 0.9, 'vmine' : 12, 'vmaxe' : 18,
            've' : 1.2 }
model.add('Mg II', **feature)

feature = { 'an' : 14, 'ai' : 1, 'tau1' : 4.5, 'vmine' : 10.0, 'vmaxe' : 16,
            've' : 1.9, 'temp' : 15 }
model.add('Si II', **feature)

feature = { 'an' : 14, 'ai' : 2, 'tau1' : 1.4, 'vmine' : 10, 'vmaxe' : 40,
            've' : 1.0 }
model.add('Si III', **feature)

feature = { 'an' : 16, 'ai' : 1, 'tau1' : 1.6, 'vmine' : 10, 'vmaxe' : 40,
            've' : 1.23 }
model.add('S II', **feature)

feature = { 'an' : 20, 'ai' : 1, 'tau1' : 70.0, 'vmine' : 12, 'vmaxe' : 18,
            've' : 1.2 }
model.add('Ca II', **feature)

feature = { 'an' : 20, 'ai' : 1, 'tau1' : 1.5, 'vmine' : 20, 'vmaxe' : 40,
            've' : 2.0 }
model.add('Ca II HV', **feature)

feature = { 'an' : 26, 'ai' : 1, 'tau1' : 0.3, 'vmine' : 10, 'vmaxe' : 40,
            've' : 1.0 }
model.add('Fe III', **feature)

feature = { 'an' : 26, 'ai' : 1, 'tau1' : 0.4, 'vmine' : 16, 'vmaxe' : 40 }
model.add('Fe II HV', **feature)

feature = { 'an' : 27, 'ai' : 1, 'tau1' : 0.2, 'vmine' : 12, 'vmaxe' : 40 }
model.add('Co II', **feature)

# feature = { 'an' : 28, 'ai' : 1, 'tau1' : 0.0, 'vmine' : 12, 'vmaxe' : 18 }
# model.add('Ni II', **feature)

# feature = { 'an' : 26, 'ai' : 1, 'tau1' : 0.0, 'vmine' : 20, 'vmaxe' : 40 }
# model.add('Ni II HV', **feature)

# feature = { 'an' : 6, 'ai' : 1, 'tau1' : 0.0, 'vmine' : 12, 'vmaxe' : 40 }
# model.add('C II', **feature)

# feature = { 'an' : 11, 'ai' : 0, 'tau1' : 0.0, 'vmine' : 12, 'vmaxe' : 40 }
# model.add('Na I', **feature)

# feature = { 'an' : 12, 'ai' : 1, 'tau1' : 0.0, 'vmine' : 20, 'vmaxe' : 40 }
# model.add('Mg II HV', **feature)

# feature = { 'an' : 14, 'ai' : 1, 'tau1' : 1.0, 'vmine' : 22, 'vmaxe' : 40 }
# model.add('Si II HV', **feature)

# feature = { 'an' : 22, 'ai' : 1, 'tau1' : 0.0, 'vmine' : 12, 'vmaxe' : 40 }
# model.add('Ti II', **feature)

# feature = { 'an' : 26, 'ai' : 1, 'tau1' : 1.0, 'vmine' : 12, 'vmaxe' : 18 }
# model.add('Fe II', **feature)


# Fitting
obs_data = read_data(fn, z=z)

# Can fit the params (feat_params) that allow synow to fit the data best.
# Bounds (feat_bounds) are for each different parameter. The feature is just
# the label from above (whatever you define it as; synow doesn't depend on it.)
# The parameters (feat_params) should be the exact synow feature parameter
# names.
bounds = [(0., 13.), (0.5, 3.0)]
model.fit(obs_data, wave_range=(5750., 6250.), feature='Si II',
          feat_params=['tau1', 've'], feat_bounds=bounds)


# Get final save
model.save()
