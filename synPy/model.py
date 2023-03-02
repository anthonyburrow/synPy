import numpy as np
from spextractor import Spextractor
from scipy.optimize import curve_fit
import subprocess

from .io import read_data


_temp_run_script = 'runsynow_temp.sh'
_temp_synthetic_fn = 'temp.dat'

_default_params = {
    'synow_lines_path'     : '$HOME/synow/lines/',
    'kurucz_linelist_path' : '$HOME/synow/kurucz_lines/',
    'refdata_path'         : '$HOME/synow/src/',
    'vphot'                :  12000.0,
    'vmax'                 :  40000.0,
    'tbb'                  :  15000.0,
    'ea'                   :  4000.0,
    'eb'                   :  8000.0,
    'nlam'                 :  1000,
    'flambda'              :  '.true.',
    'taumin'               :  0.01,
    'grid'                 :  32,
    'zeta'                 :  1.0,
    'stspec'               :  3500.0,
    'numref'               :  1,
    'delta_v'              :  300.0,
    'spectrum_file'        : 'synthetic.dat',
    'debug_out'            : '.true.',
    'do_locnorm'           : '.true.',
}

_default_feature = {
    'an'       :   14,   # Si II just because (usually overwritten anyway)
    'ai'       :    1,
    'tau1'     :  0.0,
    'pwrlawin' :  2.0,
    'vmine'    : _default_params['vphot'],
    'vmaxe'    : _default_params['vmax'],
    've'       :  1.0,
    'vmaxg'    : 12.0,
    'sigma_v'  :  2.0,
    'temp'     : 10.0,
    'dprof'    :  'e',
}


class SynowModel:

    def __init__(self, *args, **kwargs):
        '''Initialize with Spextractor read arguments'''
        self._params = dict(_default_params)
        self._features = {}

        self._fit_counter = 1

        # Setup GPR model for fitting
        self._spex = self._setup_spex(*args, **kwargs)
        self.data = self._spex.data

    @property
    def params(self):
        return self._params

    @property
    def features(self):
        return self._features

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self._params[key] = value

            if key == 'vphot':
                _default_feature['vmine'] = value / 1000.
            elif key == 'vmax':
                _default_feature['vmaxe'] = value / 1000.

    def add(self, label: str, **kwargs):
        if label not in self._features:
            self._features[label] = dict(_default_feature)

        for key, value in kwargs.items():
            self._features[label][key] = value

    def fit(self, wave_range: tuple, feature: str, feat_params: list,
            feat_bounds: list):
        print(f'\nAttempting to fit {feature}...')

        if wave_range[0] < self._spex.wave[0] or \
           wave_range[1] > self._spex.wave[-1]:
            print('WARNING: Attempting to fit outside range of observation')

        # Gets the same wavelength points as initial synow model
        wave_fit = self._run_synow_and_retrieve()[:, 0]
        wave_mask = (wave_range[0] <= wave_fit) & (wave_fit <= wave_range[1])
        wave_fit = wave_fit[wave_mask]

        obs_flux, _ = self._spex.predict(wave_fit)

        def _fit_function(wave_fit, *args):
            ''' Wrapper function to align keys + arguments and rerun synow '''
            # Change parameters
            new_params = {name: arg for name, arg in zip(feat_params, args)}
            self.add(feature, **new_params)
            print(f'iter = {self._fit_counter} | {new_params}')

            # Rerun synow
            self._run_synow(_temp_synthetic_fn, temp=True)
            model_data = read_data(_temp_synthetic_fn)
            model_data = model_data[wave_mask]
            model_flux = model_data[:, 1]

            # 1. check that model_data[:, 0] == wave_fit
            # 2. try with only Si II in model

            self._fit_counter += 1

            return model_flux

        p0 = [self._features[feature][param] for param in feat_params]
        bound_lower = [f[0] for f in feat_bounds]
        bound_upper = [f[1] for f in feat_bounds]
        bounds = (bound_lower, bound_upper)
        params, cov = curve_fit(_fit_function, wave_fit, obs_flux,
                                p0=p0, bounds=bounds)

        params_txt = {name: value for name, value in zip(feat_params, params)}
        print(f'\nUpdated new model parameters: {params_txt}\n')

        self._fit_counter = 1

    def write_script(self, out_script: str = None, *args, **kwargs):
        if out_script is None:
            out_script = 'runsynow_gen.sh'

        script_str = self._generate_script_string(*args, **kwargs)

        with open(out_script, 'w') as file:
            file.write(script_str)

        return out_script

    def get_synth(self, spectrum_file=None):
        '''Runs Synow, generates and returns a current synthetic spectrum.'''
        if spectrum_file is None:
            spectrum_file = self._params['spectrum_file']
        self._run_synow(temp=False)

        return read_data(spectrum_file)

    def _setup_spex(self, *args, **kwargs):
        spex = Spextractor(*args, **kwargs)
        spex.create_model(downsampling=3)

        return spex

    def _run_synow(self, *args, **kwargs):
        out_script = self.write_script(*args, **kwargs)

        subprocess.run(f'chmod +x {out_script}', shell=True, check=True)
        subprocess.run(f'./{out_script}', shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _generate_script_string(self, temp=False, *args, **kwargs):
        script = (
            f"#!/bin/bash"
            f"\n"
        )

        if temp:
            script += f"\nSYNTH_FILE='{_temp_synthetic_fn}'"
        else:
            script += (
                f"\nSYNTH_FILE='{self._params['spectrum_file']}'"
                f"\nBACKUP_FILE='{self._params['spectrum_file']}.bak'"
                f"\n"
                f"\nif [ -e $SYNTH_FILE ];"
                f"\nthen"
                f"\n    echo 'Copying previous model to' $BACKUP_FILE"
                f"\n    cp $SYNTH_FILE $BACKUP_FILE"
                f"\nfi"
            )

        script += (
            f"\n"
            f"\n{self._params['refdata_path']}synow << EOF"
            f"\n"
            f"\n &parms"
            f"\n"
        )

        for key, value in _default_params.items():
            value_string = f"{self._params[key]}"
            if key in ('synow_lines_path', 'kurucz_linelist_path',
                       'refdata_path'):
                value_string = f"'{value_string}'"
            elif key == 'spectrum_file':
                value_string = "'$SYNTH_FILE'"
            elif key == 'numref':
                value_string = len(self._features)

            script += f"    {key.ljust(20)} = {value_string},\n"

        script += '\n'

        for key in _default_feature:
            script += f"    {key.ljust(8)} ="

            values = []
            for name, feature in self._features.items():
                value = feature[key]
                if key == 'dprof':
                    value = f"'{value}'"
                n_digits = 16
                value = str(value)[:n_digits].rjust(n_digits + 1)
                values.append(value)
            script += ','.join(values)

            script += '\n'

        script += (
            f"\n/"
            f"\n"
            f"\nEOF"
            f"\n"
        )

        return script
