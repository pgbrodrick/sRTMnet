#! /usr/bin/env python
#
#  Copyright 2020 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Author: Philip G Brodrick, philip.brodrick@jpl.nasa.gov


import numpy as np
import os
import json
import datetime
import ray
import argparse
from types import SimpleNamespace
import logging
import subprocess

from scipy.stats import qmc

from isofit.utils.template_construction import (
    write_modtran_template,
    SerialEncoder,
    get_lut_subset,
)
from isofit.data import env
from isofit.radiative_transfer.radiative_transfer import confPriority 
from isofit.radiative_transfer.engines.modtran import ModtranRT
from isofit.radiative_transfer.engines.six_s import SixSRT
from isofit.configs import configs, Config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('-dir', default='', help='Working Dir')
    parser.add_argument('-n_cores', type=int, default=1)
    parser.add_argument('-train', type=int, default=1, choices=[0,1])
    parser.add_argument('-cleanup', type=int, default=0, choices=[0,1])
    parser.add_argument('-ip_head', type=str)
    parser.add_argument('-redis_password', type=str)
    parser.add_argument(
        '--configure_and_exit', 
        default=False, 
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--sobel', 
        default=False, 
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '-sixs_path',
        default=None,
        help='SIXS Installation DIR'
    )
    parser.add_argument(
        '-modtran_path',
        default=None,
        help='MODTRAN Installation DIR'
    )
    parser.add_argument(
        '--coarse', 
        default=None, 
        type=str
    )
    args = parser.parse_args()

    args.training = args.train == 1
    args.cleanup = args.cleanup == 1

    dayofyear = 200

    paths = Paths(
        args,
        args.training
    )

    if args.sobel:
        if args.train:
            # Get bounds from regular grid
            bounds = {
                'to_solar_zenith_bnds': [0, 50],
                'to_sensor_zenith_bnds': [140, 180],
                'altitude_km_bnds': [2, 25],
                'elevation_km_bnds': [0.01, 4.5],
                'h2o_bnds': [0.1, 5],
                'aerfrac_2_bnds': [0.01, 1],
            }
            # consts won't be sampled from the grid
            consts = {
                'to_solar_azimuth_bnds': 180,
                'to_sensor_azimuth_bnds': 180,
            }
            # n is the number of samples to take from the grid.
            grid = sobel_lut(bounds, consts, n=3)

            # Add in relative azimuth
            relative_azimuth = np.abs(
                np.array(grid['to_solar_azimuth_lut'])
                - np.array(grid['to_sensor_azimuth_lut'])
            )
            grid['relative_azimuth_lut'] = [float(i) for i in np.minimum(
                relative_azimuth,
                360 - relative_azimuth
            )]

        else:
            # Get bounds from regular grid
            bounds = {
                'to_solar_zenith_bnds': [0, 50],
                'to_sensor_zenith_bnds': [140, 180],
                'altitude_km_bnds': [2, 25],
                'elevation_km_bnds': [0.01, 4.5],
                'h2o_bnds': [0.1, 5],
                'aerfrac_2_bnds': [0.01, 1],
            }
            consts = {
                'to_solar_azimuth_bnds': 180,
                'to_sensor_azimuth_bnds': 180,
            }
            grid = sobel_lut(bounds, consts, n=10)

            # Add in relative azimuth
            relative_azimuth = np.abs(
                np.array(grid['to_solar_azimuth_lut'])
                - np.array(grid['to_sensor_azimuth_lut'])
            )
            grid['relative_azimuth_lut'] = [float(i) for i in np.minimum(
                relative_azimuth,
                360 - relative_azimuth
            )]
    else:
        if args.train:
            grid = {
                #'to_solar_zenith_lut': [0, 12.5, 25, 37.5, 50],
                'to_solar_zenith_lut': [0, 30, 60],
                #'to_solar_azimuth_lut': [0, 60, 120, 180],
                'to_solar_azimuth_lut': [0, 90, 180],
                'to_sensor_azimuth_lut': [0],
                'to_sensor_zenith_lut': [140, 160, 180],
                'altitude_km_lut': [2, 4, 7, 10, 15, 25, 99],
                #'elevation_km_lut': [0, 0.75, 1.5, 2.25, 4.5, 6],
                'elevation_km_lut': [0, 1.5, 4.5, 6],
                'h2o_lut': list(np.sort(
                    np.round(np.linspace(0.1, 5, num=5), 3).tolist()
                )), #+ [0.7125, 1.9375, 3.162, 4.3875],
                'aerfrac_2_lut': list(np.sort(
                    np.round(np.linspace(0.01, 1, num=5), 3).tolist()
                ))
                #'to_solar_zenith_lut': [0, 50],
                #'to_solar_azimuth_lut': [0, 180],
                #'to_sensor_azimuth_lut': [0],
                #'to_sensor_zenith_lut': [140, 180],
                #'altitude_km_lut': [2, 99],
                #'elevation_km_lut': [0, 6],
                #'h2o_lut': list(np.sort(
                #    np.round(np.linspace(0.1, 5, num=2), 3).tolist() + [0.6125]
                #)),
                #'aerfrac_2_lut': list(np.sort(
                #    np.round(np.linspace(0.01, 1, num=5), 3).tolist() + [0.5]
                #))
            }
            relative_azimuth_lut = np.abs(
                np.array(grid['to_solar_azimuth_lut']) 
                - np.array(grid['to_sensor_azimuth_lut'])
            )
            grid['relative_azimuth_lut'] = list(np.minimum(
                relative_azimuth_lut, 
                360 - relative_azimuth_lut
            ))

        else:
            # HOLDOUT SET
            grid = {
                'to_solar_zenith_lut': [6, 18, 30,45],
                'to_solar_azimuth_lut': [135],
                'to_sensor_azimuth_lut': [0],
                'to_sensor_zenith_lut': [145, 155, 165, 175],
                'altitude_km_lut': [3, 5.5, 8.5, 12.5, 17.5],
                'elevation_km_lut': [0.325, 1.025, 1.875, 2.575, 4.2],
                'h2o_lut': np.round(np.linspace(0.5, 4.5, num=4), 3),
                'aerfrac_2_lut': np.round(np.linspace(0.125, 0.9, num=4), 3),
            }
            relative_azimuth_lut = np.abs(
                np.array(grid['to_solar_azimuth_lut']) 
                - np.array(grid['to_sensor_azimuth_lut'])
            )
            grid['relative_azimuth_lut'] = np.minimum(
                relative_azimuth_lut, 
                360 - relative_azimuth_lut
            )

    n_lut_build = np.prod([
        len(grid['to_solar_zenith_lut']),
        len(grid['to_sensor_zenith_lut']),
        len(grid['relative_azimuth_lut']),
        len(grid['altitude_km_lut']),
        len(grid['elevation_km_lut']),
        len(grid['h2o_lut']),
        len(grid['aerfrac_2_lut'])
    ])

    print('Num LUTs to build: {}'.format(n_lut_build))
    print('Expected MODTRAN runtime: {} hrs'.format(n_lut_build*1.5))
    print('Expected MODTRAN runtime: {} days'.format(n_lut_build*1.5/24))
    print('Expected MODTRAN runtime per (40-core) node: {} days'.format(n_lut_build*1.5/24/40))

    
    # Create wavelength file
    if args.coarse is None:
        wl = np.arange(0.350, 2.550, 0.0001)
        wl_file_contents = np.zeros((len(wl),3))
        wl_file_contents[:,0] = np.arange(len(wl),dtype=int)
        wl_file_contents[:,1] = wl
        wl_file_contents[:,2] = 0.0001

        np.savetxt(
            paths.wavelength_file, 
            wl_file_contents,fmt='%.5f'
        )

    write_modtran_template(
        atmosphere_type='ATM_MIDLAT_SUMMER', 
        fid=os.path.splitext(paths.modtran_template_file)[0],
        altitude_km=grid['altitude_km_lut'][0],
        dayofyear=dayofyear, 
        to_sensor_azimuth=grid['to_sensor_azimuth_lut'][0],
        to_sensor_zenith=grid['to_sensor_zenith_lut'][0],
        to_sun_zenith=grid['to_solar_zenith_lut'][0],
        relative_azimuth=grid['relative_azimuth_lut'][0],
        gmtime=0, 
        elevation_km=grid['elevation_km_lut'][0], 
        output_file=paths.modtran_template_file,
        ihaze_type="AER_NONE",
    )

    # Make sure H2O grid is fully valid
    with open(paths.modtran_template_file, 'r') as f:
        modtran_config = json.load(f)

    for n in range(len(modtran_config['MODTRAN'])):
        modtran_config['MODTRAN'][n]['MODTRANINPUT']['GEOMETRY']['IPARM'] = 12
        modtran_config['MODTRAN'][n]['MODTRANINPUT']['ATMOSPHERE']['H2OOPT'] = '+'
        modtran_config['MODTRAN'][n]['MODTRANINPUT']['AEROSOLS']['VIS'] = 100
        if args.coarse is not None:
            modtran_config['MODTRAN'][n]['MODTRANINPUT']['SPECTRAL']['BMNAME'] = '05_2013'
            modtran_config['MODTRAN'][n]['MODTRANINPUT']['SPECTRAL']['DV'] = 5
            modtran_config['MODTRAN'][n]['MODTRANINPUT']['SPECTRAL']['FWHM'] = 5
            
    with open(paths.modtran_template_file, 'w') as fout:
        fout.write(json.dumps(
            modtran_config, 
            cls=SerialEncoder, 
            indent=4, 
            sort_keys=True
        ))

    configure_and_exit = args.configure_and_exit
    build_main_config(
        paths, 
        grid['h2o_lut'], 
        grid['elevation_km_lut'], 
        grid['altitude_km_lut'], 
        grid['to_sensor_zenith_lut'], 
        grid['to_solar_zenith_lut'], 
        grid['to_sensor_azimuth_lut'], 
        grid['to_solar_azimuth_lut'], 
        grid['relative_azimuth_lut'],
        grid['aerfrac_2_lut'], 
        paths.isofit_config_file, 
        configure_and_exit=configure_and_exit,
        ip_head=args.ip_head,
        redis_password=args.redis_password,
        n_cores=args.n_cores
    )

    config = configs.create_new_config(paths.isofit_config_file)

    if args.configure_and_exit:
        ray.init(
            num_cpus=args.n_cores,
            _temp_dir=config.implementation.ray_temp_dir ,
            include_dashboard=False,
            local_mode=args.n_cores == 1,
        )

    else:
        # Initialize ray for parallel execution
        rayargs = {
            'address': config.implementation.ip_head,
            '_redis_password': config.implementation.redis_password,
            'local_mode': args.n_cores == 1
        }

        if args.n_cores < 40:
            rayargs['num_cpus'] = args.n_cores

        ray.init(**rayargs)
        print(ray.cluster_resources())

    # Inits of engines based on radiative_transfer.py
    rt_config = config.forward_model.radiative_transfer
    instrument_config = config.forward_model.instrument
    lut_grid_config = rt_config.lut_grid

    _keys = [
        "interpolator_style",
        "overwrite_interpolator",
        "lut_grid",
        "lut_path",
        "wavelength_file",
    ]

    modtran_engine_config = rt_config.radiative_transfer_engines[0]
    params = {
        key: confPriority(key, [
            modtran_engine_config, instrument_config, rt_config
        ]) 
        for key in _keys
    }
    params["engine_config"] = modtran_engine_config 
    params["build_interpolators"] = False 
    params['lut_grid'] = {
        key: params['lut_grid'][key] for key in
        modtran_engine_config.lut_names.keys()
    }
    isofit_modtran = ModtranRT(**params)

    sixs_engine_config = rt_config.radiative_transfer_engines[1]
    params = {
        key: confPriority(key, [
            sixs_engine_config, instrument_config, rt_config
        ])
        for key in _keys
    }
    params["engine_config"] = sixs_engine_config 
    params["build_interpolators"] = False 
    params['lut_grid'] = {
        key: params['lut_grid'][key] for key in
        sixs_engine_config.lut_names.keys()
    }
    # params['modtran_emulation'] = True

    # always overwrite 6S wavelength with full raw 6S range
    sixs_wl = np.arange(350, 2500 + 2.5, 2.5)
    sixs_fwhm = np.full(sixs_wl.size, 2.0)
    params['wl'] = sixs_wl
    params['fwhm'] = sixs_fwhm
    isofit_sixs = SixSRT(**params)

    # cleanup
    if args.cleanup:
        for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
            cmd = 'find {os.path.join(paths.lut_modtran_directory)} -name "{to_rm}"'
            print(cmd)
            os.system(cmd)


def build_modtran_configs(isofit_config: Config, template_file: str):
    with open(template_file, 'r') as f:
       modtran_config = json.load(f)


class Paths():

    def __init__(self, args, training=True):

        working_dir = args.dir

        self.support_dir = os.path.join(working_dir, 'support')
        self.template_dir = os.path.join(working_dir, 'templates')

        # Make dirs
        os.makedirs(self.support_dir, exist_ok=True)
        os.makedirs(self.template_dir, exist_ok=True)

        self.modtran_template_file = os.path.join(
            self.template_dir, 'modtran_template.json'
        )

        if training:
            self.isofit_config_file = os.path.join(
                self.template_dir, 'isofit_template_v2.json'
            )
        else:
            self.isofit_config_file = os.path.join(
                self.template_dir, 'isofit_template_holdout.json'
            )

        self.aerosol_tpl_path =  os.path.join(
            self.support_dir, 
            'aerosol_template.json'
        )
        self.aerosol_model_path =  os.path.join(
            self.support_dir,
            'aerosol_model.txt'
        )

        if args.coarse:
            self.wavelength_file = os.path.join(
                self.support_dir,
                'coarse_wavelengths.txt'
            )
            if os.path.isfile(args.coarse) is False:
               print(f'No wavelength file found for args.coarse at {args.coarse}') 
               exit()
            subprocess.call(f'cp {args.coarse} {self.wavelength_file}',shell=True)
        else:
            self.wavelength_file = os.path.join(
                self.support_dir,
                'hifidelity_wavelengths.txt'
            )

        # If we want to format this with the data repo
        # self.earth_sun_distance_file = str(env.path("data", "emit_noise.txt")
        self.earth_sun_distance_file = os.path.join(
            self.support_dir,
            'earth_sun_distance.txt'
        )
        self.irradiance_file = os.path.join(
            self.support_dir,
            'kurudz_0.1nm.dat'
        )

        if training:
            self.lut_modtran_directory = os.path.join(
                working_dir,
                'modtran_lut'
            )
            self.lut_sixs_directory = os.path.join(
                working_dir,
                'sixs_lut'
            )
        else:
            self.lut_modtran_directory = os.path.join(
                working_dir,
                'modtran_lut_holdout_az'
            )
            self.lut_sixs_directory = os.path.join(
                working_dir,
                'sixs_lut_holdout_az'
            )

        os.makedirs(self.lut_modtran_directory, exist_ok=True)
        os.makedirs(self.lut_sixs_directory, exist_ok=True)

        # RTE Installation
        if args.modtran_path:
            self.modtran_path = args.modtran_path
        else:
            self.modtran_path = os.getenv("MODTRAN_DIR", env.modtran)

        if args.sixs_path:
            self.sixs_path = args.sixs_path
        else:
            self.sixs_path = os.getenv("SIXS_DIR", env.sixs)


def build_main_config(
    paths, 
    h2o_lut_grid: np.array,
    elevation_lut_grid: np.array,
    altitude_lut_grid: np.array, 
    to_sensor_zenith_lut_grid: np.array,
    to_solar_zenith_lut_grid: np.array,
    to_sensor_azimuth_lut_grid: np.array,
    to_solar_azimuth_lut_grid: np.array,
    relative_azimuth_lut_grid: np.array,
    aerfrac_2_lut_grid: np.array,
    config_output_path, 
    configure_and_exit: bool = True,
    ip_head: str = None,
    redis_password: str = None,
    n_cores: int = 1,
):
    """ Write an isofit dummy config file, so we can pass in for luts.

    Args:
        paths:                      Object containing references to all relevant file locations
        lut_params:                 Configuration parameters for the lut grid
        h2o_lut_grid:               The water vapor look up table grid isofit should use for this solve
        elevation_lut_grid:         The ground elevation look up table grid isofit should use for this solve
        altitude_lut_grid:          The altitude elevation (km) look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid:  The to-sensor zenith angle look up table grid isofit should use for this solve
        to_solar_zenith_lut_grid:   The to-sun zenith angle look up table grid isofit should use for this solve
        relative_azimuth_lut_grid:  The relative to-sun azimuth angle look up table grid isofit should use for
        config_output_path:         Path to write config to
        n_cores:                    The number of cores to use during processing

    """

    # Initialize the RT config with the engines portion.
    radiative_transfer_config = {
            "radiative_transfer_engines": {
                "modtran": {
                    "engine_name": 'modtran',
                    "sim_path": paths.lut_modtran_directory,
                    "lut_path": os.path.join(
                        paths.lut_modtran_directory,
                        'lut.nc'
                    ),
                    "rt_mode": 'rdn',
                    "irradiance_file": paths.irradiance_file,
                    "multipart_transmittance": True,
                    "template_file": paths.modtran_template_file,
                    "rte_configure_and_exit": configure_and_exit,
                    "engine_base_dir": paths.modtran_path,
                    #lut_names - populated below
                    #statevector_names - populated below
                },
                "sixs": {
                    "engine_name": '6s',
                    "rt_mode": 'rdn',
                    "sim_path": paths.lut_sixs_directory,
                    "lut_path": os.path.join(
                        paths.lut_sixs_directory,
                        'lut.nc'
                    ),
                    "irradiance_file": paths.irradiance_file,
                    "earth_sun_distance_file": paths.earth_sun_distance_file,
                    "month": 6, # irrelevant to readouts we care about
                    "day": 1, # irrelevant to readouts we care about
                    "elev": elevation_lut_grid[0],
                    "alt": altitude_lut_grid[0],
                    "viewaz": to_sensor_azimuth_lut_grid[0],
                    "viewzen": 180 - to_sensor_zenith_lut_grid[0],
                    "solaz": to_solar_azimuth_lut_grid[0],
                    "solzen": to_solar_zenith_lut_grid[0],
                    "multipart_transmittance": True,
                    "template_file": paths.modtran_template_file,
                    "rte_configure_and_exit": configure_and_exit,
                    "engine_base_dir": paths.sixs_path,
                    # lut_names - populated below
                    # statevector_names - populated below
                }
            },
            "lut_grid": {},
            "unknowns": {
                "H2O_ABSCO": 0.0
            }
    }

    # Question to figure out: Do I need the second key on a lot of these?
    # H2O LUT Grid
    if h2o_lut_grid is not None and len(h2o_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['H2OSTR'] = [
            max(0.0, float(q)) for q in h2o_lut_grid
        ]

    # Elevation LUT Grid
    if elevation_lut_grid is not None and len(elevation_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['surface_elevation_km'] = [
            max(0.0, float(q)) for q in elevation_lut_grid
        ]

    # Altitude LUT Grid
    if altitude_lut_grid is not None and len(altitude_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['observer_altitude_km'] = [
            max(0.0, float(q)) for q in altitude_lut_grid
        ]
        # radiative_transfer_config['lut_grid']['alt'] = [
        #     float(q) for q in altitude_lut_grid
        # ]

    # Observer Zenith LUT Grid
    if (
        to_sensor_zenith_lut_grid is not None 
        and len(to_sensor_zenith_lut_grid) > 1
    ):
        # Does Sixs? automatically convert value from Modtran convention
        # Don't think so. Happens in template_construction.py
        # Do I have to carry these as two keys?
        # radiative_transfer_config['lut_grid']['observer_zenith'] = [
        #     float(q) for q in to_sensor_zenith_lut_grid
        # ]
        # Sixs convension
        # radiative_transfer_config['lut_grid']['viewzen'] = [
        radiative_transfer_config['lut_grid']['observer_zenith'] = [
            180 - float(q) for q in to_sensor_zenith_lut_grid
        ] 

    # Solar  Zenith LUT Grid
    if (
        to_solar_zenith_lut_grid is not None 
        and len(to_solar_zenith_lut_grid) > 1
    ):
        # Modtran convension
        radiative_transfer_config['lut_grid']['solar_zenith'] = [
            float(q) for q in to_solar_zenith_lut_grid
        ]

    # Azimuth LUT Grid
    if (
        relative_azimuth_lut_grid is not None 
        and len(relative_azimuth_lut_grid) > 1
    ):
        radiative_transfer_config["lut_grid"]["relative_azimuth"] = [
            float(q) for q in relative_azimuth_lut_grid
        ]

    # Aerosol LUT Grid
    # Should be able to use AERFRAC_2 for both
    if len(aerfrac_2_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['AERFRAC_2'] = [
            float(q) for q in aerfrac_2_lut_grid
        ]
        radiative_transfer_config['lut_grid']['AOT550'] = [
            float(q) for q in aerfrac_2_lut_grid
        ]

    if paths.aerosol_model_path is not None:
        radiative_transfer_config[
            'radiative_transfer_engines'
        ]['modtran']['aerosol_model_file'] = paths.aerosol_model_path

    if paths.aerosol_tpl_path is not None:
        radiative_transfer_config[
            'radiative_transfer_engines'
        ]['modtran']["aerosol_template_file"] = paths.aerosol_tpl_path

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    # Populate the lut_names and the statevector_names
    modtran_lut_names = {
        x: None for x in [
            'H2OSTR','surface_elevation_km','observer_altitude_km',
            'observer_zenith','solar_zenith', 'relative_azimuth',
            'AERFRAC_2'
        ] if x in radiative_transfer_config['lut_grid'].keys()
    }
    radiative_transfer_config[
        'radiative_transfer_engines'
    ]['modtran']['lut_names'] = modtran_lut_names
    radiative_transfer_config["radiative_transfer_engines"]["modtran"][
        "statevector_names"] = list(modtran_lut_names.keys())

    sixs_lut_names = {
        x: None for x in [
            'H2OSTR','surface_elevation_km','observer_altitude_km',
            # 'viewzen','solar_zenith', 'relative_azimuth',
            'observer_zenith','solar_zenith', 'relative_azimuth',
            'AOT550'
        ] if x in radiative_transfer_config['lut_grid'].keys()
    }
    radiative_transfer_config[
        'radiative_transfer_engines'
    ]['sixs']['lut_names'] = sixs_lut_names
    radiative_transfer_config["radiative_transfer_engines"]["sixs"][
        "statevector_names"
    ] = list(sixs_lut_names.keys())

    # Inversion windows - Not sure what to use here
    # inversion_windows = [[1,2],[3,4]]
    inversion_windows = [[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]]

    # make isofit configuration
    isofit_config_modtran = {
        'input': {},
        'output': {}, 
        'forward_model': {
            "radiative_transfer": radiative_transfer_config,
            "instrument": {"wavelength_file": paths.wavelength_file}
        },
        "implementation": {
            "inversion": {"windows": inversion_windows},
            "n_cores": n_cores,
            "ip_head": ip_head,
            "redis_password": redis_password
        }
    }
    isofit_config_modtran['implementation']["rte_configure_and_exit"] = True

    # write modtran_template
    with open(config_output_path, 'w') as fout:
        fout.write(json.dumps(
            isofit_config_modtran, 
            cls=SerialEncoder, 
            indent=4, 
            sort_keys=True
        ))


def sobel_lut(bounds, consts={}, n=3, log=False):
    """
    Create a look using a Sobel sequence

    Args:
        bounds (dict): keys are the lut element names. Values (list) are the bounds (idx 0 and 1) of the sequence. idx 2 is the number of samples to take
        ns (list): Number of samples to take from sequence for each bounds element. The length of bounds
                  should math the length of n
        consts (dict): keys are the lut element names. Values are the constant values to use
        log (bool): Flag for log transform before sampling
    Returns:
        grid (dict): lut grid.
    """

    # Handle cos transform
    cosd = lambda a : np.cos(np.deg2rad(a))
    cos_keys = [
        'to_solar_zenith_bnds',
        'to_sensor_zenith_bnds',
        'to_solar_azimuth_bnds',
        'to_sensor_azimuth_bnds',
    ]
    for key, bnd in bounds.items():
        if key in cos_keys:
            bnd = [cosd(bnd[0]), cosd(bnd[1])]
            bounds[key] = bnd

    # Handle input log transform
    log_keys = [
        'altitude_km_bnds',
        'elevation_km_bnds',
        'h2o_bnds',
        'aerfrac_2_bnds',
    ]
    if log:
        for key, bnd in bounds.items():
            if key in log_keys:
                bnd = [np.log(bnd[0]), np.log(bnd[1])]
                bounds[key] = bnd

    # Sort bounds
    for key, bnd in bounds.items():
        bounds[key] = sorted(bnd)

    l_bound = [bnd[0] for bnd in bounds.values()]
    r_bound = [bnd[1] for bnd in bounds.values()]

    # Make the sobel sequence
    # Not 100% sure if we need to scamble the sequence gen
    sobol = qmc.Sobol(d=len(bounds), scramble=False)
    seq = sobol.random(n=n)
    seq = qmc.scale(
        seq, 
        l_bounds=l_bound, 
        u_bounds=r_bound
    )

    # Transform sample out of cos
    arccosd = lambda a : np.rad2deg(np.arccos(a))
    cos_i = [i for i, key in enumerate(bounds.keys()) if key in cos_keys] 
    seq[:, cos_i] = arccosd(seq[:, cos_i])

    # Transform out of log space
    if log:
        log_i = [i for i, key in enumerate(bounds.keys()) if key in log_keys] 
        seq[:, log_i] = np.exp(seq[:, log_i])

    # Construct the grid-dict
    grid = {}
    for i, key in enumerate(bounds.keys()):
        key_lut = '_'.join(key.split('_')[:-1] + ['lut'])
        grid[key_lut] = [float(i) for i in seq[:, i]]

    # Add the constants onto the grid 
    for key, value in consts.items():
        key_lut = '_'.join(key.split('_')[:-1] + ['lut'])
        grid[key_lut] = [float(value)]

    # Trim out non-physical components 
    # (e.g., elevation > observation altitude)
    # The best way I can do this with uneven clolumns is to remove 
    # alt that are below max elev
    grid['altitude_km_lut'] = np.array(grid['altitude_km_lut'])[
        grid['altitude_km_lut'] > np.max(grid['elevation_km_lut'])
    ]

    # Do some cleaning a sorting 
    for key, value in grid.items():
        grid[key] = np.round(sorted(value), 4)

    return grid


if __name__ == '__main__':
    main()
