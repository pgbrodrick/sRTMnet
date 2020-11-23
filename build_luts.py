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
from isofit.utils.apply_oe import write_modtran_template, SerialEncoder
import os
import json
from isofit.radiative_transfer.modtran import ModtranRT
from isofit.radiative_transfer.six_s import SixSRT
from isofit.configs import configs, Config
import datetime
import ray
import argparse
from isofit.core.sunposition import sunpos


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('-ip_head', type=str)
    parser.add_argument('-redis_password', type=str)
    parser.add_argument('-n_cores', type=int, default=1)
    parser.add_argument('-train', type=int, default=1, choices=[0,1])
    parser.add_argument('-cleanup', type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    args.train = args.train == 1
    args.cleanup = args.cleanup == 1

    dayofyear = 200

    if args.train:
        to_solar_zenith_lut = [0, 12.5, 25, 37.5, 50]
        to_solar_azimuth_lut = [180]
        to_sensor_azimuth_lut = [180]
        to_sensor_zenith_lut = [140, 160, 180]
        altitude_km_lut = [2, 4, 7, 10, 15, 25]
        elevation_km_lut = [0, 0.75, 1.5, 2.25, 4.5]
        h2o_lut_grid = np.round(np.linspace(0.1,5,num=5),3).tolist() + [0.6125]
        h2o_lut_grid.sort()
        aerfrac_2_lut_grid = np.round(np.linspace(0.01,1,num=5),3).tolist() + [0.5]
        aerfrac_2_lut_grid.sort()

    else:
        # HOLDOUT SET
        to_solar_zenith_lut = [6, 18, 30,45]
        to_solar_azimuth_lut = [60, 300]
        to_sensor_azimuth_lut = [180]
        to_sensor_zenith_lut = [145, 155, 165, 175]
        altitude_km_lut = [3, 5.5, 8.5, 12.5, 17.5]
        elevation_km_lut = [0.325, 1.025, 1.875, 2.575, 4.2]
        h2o_lut_grid = np.round(np.linspace(0.5,4.5,num=4),3)
        aerfrac_2_lut_grid = np.round(np.linspace(0.125,0.9,num=4),3)



    n_lut_build = np.product([len(to_solar_zenith_lut),
                                                     len(to_solar_azimuth_lut),
                                                     len(to_sensor_zenith_lut),
                                                     len(to_sensor_azimuth_lut),
                                                     len(altitude_km_lut),
                                                     len(elevation_km_lut),
                                                     len(h2o_lut_grid),
                                                     len(aerfrac_2_lut_grid)])

    print('Num LUTs to build: {}'.format(n_lut_build))
    print('Expected MODTRAN runtime: {} hrs'.format(n_lut_build*1.5))
    print('Expected MODTRAN runtime: {} days'.format(n_lut_build*1.5/24))
    print('Expected MODTRAN runtime per (40-core) node: {} days'.format(n_lut_build*1.5/24/40))
    
    # Create wavelength file
    wl = np.arange(0.350, 2.550, 0.0005)
    wl_file_contents = np.zeros((len(wl),3))
    wl_file_contents[:,0] = np.arange(len(wl),dtype=int)
    wl_file_contents[:,1] = wl
    wl_file_contents[:,2] = 0.0005
    np.savetxt('support/hifidelity_wavelengths.txt',wl_file_contents,fmt='%.5f')

    # Initialize ray for parallel execution
    rayargs = {'address': args.ip_head,
               'redis_password': args.redis_password,
               'local_mode': args.n_cores == 1}

    if args.n_cores < 40:
        rayargs['num_cpus'] = args.n_cores
    ray.init(**rayargs)
    print(ray.cluster_resources())

    template_dir = 'templates'
    if os.path.isdir(template_dir) is False:
        os.mkdir(template_dir)

    modtran_template_file = os.path.join(template_dir,'modtran_template.json')

    if args.training:
        isofit_config_file = os.path.join(template_dir,'isofit_template_v2.json')
    else:
        isofit_config_file = os.path.join(template_dir,'isofit_template_holdout.json')

    write_modtran_template(atmosphere_type='ATM_MIDLAT_SUMMER', fid=os.path.splitext(modtran_template_file)[0],
                           altitude_km=altitude_km_lut[0],
                          dayofyear=dayofyear, latitude=to_solar_azimuth_lut[0], longitude=to_solar_zenith_lut[0],
                          to_sensor_azimuth=to_sensor_azimuth_lut[0], to_sensor_zenith=to_sensor_zenith_lut[0],
                          gmtime=0, elevation_km=elevation_km_lut[0], output_file=modtran_template_file)


    # Make sure H2O grid is fully valid
    with open(modtran_template_file, 'r') as f:
        modtran_config = json.load(f)
    modtran_config['MODTRAN'][0]['MODTRANINPUT']['GEOMETRY']['IPARM'] = 12
    modtran_config['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['H2OOPT'] = '+'
    modtran_config['MODTRAN'][0]['MODTRANINPUT']['AEROSOLS']['VIS'] = 100
    with open(modtran_template_file, 'w') as fout:
        fout.write(json.dumps(modtran_config, cls=SerialEncoder, indent=4, sort_keys=True))

    paths = Paths(os.path.join('.',os.path.basename(modtran_template_file)), args.training)


    build_main_config(paths, isofit_config_file, to_solar_azimuth_lut, to_solar_zenith_lut, 
                      aerfrac_2_lut_grid, h2o_lut_grid, elevation_km_lut, altitude_km_lut, to_sensor_azimuth_lut,
                      to_sensor_zenith_lut, n_cores=args.n_cores)

    config = configs.create_new_config(isofit_config_file)


    isofit_modtran = ModtranRT(config.forward_model.radiative_transfer.radiative_transfer_engines[0],
                               config)

    isofit_sixs = SixSRT(config.forward_model.radiative_transfer.radiative_transfer_engines[1],
                         config)

    # cleanup
    if args.cleanup:
        for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
            cmd = 'find {os.path.join(paths.lut_modtran_directory)} -name "{to_rm}"')
            print(cmd)
            os.system(cmd)



def build_modtran_configs(isofit_config: Config, template_file: str):
    with open(template_file, 'r') as f:
       modtran_config = json.load(f)


class Paths():

    def __init__(self, modtran_tpl, training=True):
        self.aerosol_tpl_path =  '../support/aerosol_template.json'
        self.aerosol_model_path =  '../support/aerosol_model.txt'
        self.wavelenth_file = '../support/hifidelity_wavelengths.txt'
        self.earth_sun_distance_file = '../support/earth_sun_distance.txt'
        self.irradiance_file = '../support/prism_optimized_irr.dat'

        if args.training:
            self.lut_modtran_directory = '../modtran_lut'
            self.lut_sixs_directory = '../sixs_lut'
        else:
            self.lut_modtran_directory = '../modtran_lut_holdout_az'
            self.lut_sixs_directory = '../sixs_lut_holdout_az'

        self.modtran_template_path = modtran_tpl


def build_main_config(paths, config_output_path, to_solar_azimuth_lut_grid: np.array, to_solar_zenith_lut_grid: np.array, 
                      aerfrac_2_lut_grid: np.array, h2o_lut_grid: np.array = None,
                      elevation_lut_grid: np.array = None, altitude_lut_grid: np.array = None, to_sensor_azimuth_lut_grid: np.array = None,
                      to_sensor_zenith_lut_grid: np.array = None, n_cores: int = 1):
    """ Write an isofit dummy config file, so we can pass in for luts.

    Args:
        paths: object with relevant path information attatched
        config_output_path: path to write config to
        to_solar_azimuth_lut_grid: the to-solar azimuth angle look up table grid to build
        to_solar_zenith_lut_grid: the to-solar zenith angle look up table grid to build
        aerfrac_2_lut_grid: the aerosol 2 look up table grid isofit should use for this solve
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        elevation_lut_grid: the ground elevation look up table grid isofit should use for this solve
        altitude_lut_grid: the acquisition altitude look up table grid isofit should use for this solve
        to_sensor_azimuth_lut_grid: the to-sensor azimuth angle look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid: the to-sensor zenith angle look up table grid isofit should use for this solve
        n_cores: the number of cores to use during processing

    """


    radiative_transfer_config = {

            "radiative_transfer_engines": {
                "modtran": {
                    "engine_name": 'modtran',
                    "lut_path": paths.lut_modtran_directory,
                    "template_file": paths.modtran_template_path,
                    "wavelength_range": [350,2500],
                    #lut_names - populated below
                    #statevector_names - populated below
                },
                "sixs": {
                    "engine_name": '6s',
                    "lut_path": paths.lut_sixs_directory,
                    "wavelength_range": [350, 2500],
                    "irradiance_file": paths.irradiance_file,
                    "earth_sun_distance_file": paths.earth_sun_distance_file,
                    "month": 6, # irrelevant to readouts we care about
                    "day": 1, # irrelevant to readouts we care about
                    "elev": elevation_lut_grid[0],
                    "alt": altitude_lut_grid[0],
                    "viewaz": to_sensor_azimuth_lut_grid[0],
                    "viewzen": 180 - to_sensor_zenith_lut_grid[0],
                    "solaz": to_solar_azimuth_lut_grid[0],
                    "solzen": to_solar_zenith_lut_grid[0]
                    # lut_names - populated below
                    # statevector_names - populated below
                }
            },
            "lut_grid": {},
            "unknowns": {
                "H2O_ABSCO": 0.0
            }
    }
    if h2o_lut_grid is not None and len(h2o_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['H2OSTR'] = [max(0.0, float(q)) for q in h2o_lut_grid]

    if elevation_lut_grid is not None and len(elevation_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['GNDALT'] = [max(0.0, float(q)) for q in elevation_lut_grid]
        radiative_transfer_config['lut_grid']['elev'] = [float(q) for q in elevation_lut_grid]

    if altitude_lut_grid is not None and len(altitude_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['H1ALT'] = [max(0.0, float(q)) for q in altitude_lut_grid]
        radiative_transfer_config['lut_grid']['alt'] = [float(q) for q in altitude_lut_grid]

    if to_sensor_azimuth_lut_grid is not None and len(to_sensor_azimuth_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['TRUEAZ'] = [float(q) for q in to_sensor_azimuth_lut_grid]
        radiative_transfer_config['lut_grid']['viewaz'] = [float(q) for q in to_sensor_azimuth_lut_grid]

    if to_sensor_zenith_lut_grid is not None and len(to_sensor_zenith_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['OBSZEN'] = [float(q) for q in to_sensor_zenith_lut_grid] # modtran convension
        radiative_transfer_config['lut_grid']['viewzen'] = [180 - float(q) for q in to_sensor_zenith_lut_grid] # sixs convension

    if to_solar_zenith_lut_grid is not None and len(to_solar_zenith_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['solzen'] = [float(q) for q in to_solar_zenith_lut_grid] # modtran convension

    if to_solar_azimuth_lut_grid is not None and len(to_solar_azimuth_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['solaz'] = [float(q) for q in to_solar_azimuth_lut_grid] # modtran convension

    # add aerosol elements from climatology
    if len(aerfrac_2_lut_grid) > 1:
        radiative_transfer_config['lut_grid']['AERFRAC_2'] = [float(q) for q in aerfrac_2_lut_grid]
        radiative_transfer_config['lut_grid']['AOT550'] = [float(q) for q in aerfrac_2_lut_grid]

    if paths.aerosol_model_path is not None:
        radiative_transfer_config['radiative_transfer_engines']['modtran']['aerosol_model_file'] = paths.aerosol_model_path
    if paths.aerosol_tpl_path is not None:
        radiative_transfer_config['radiative_transfer_engines']['modtran']["aerosol_template_file"] = paths.aerosol_tpl_path

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    radiative_transfer_config['radiative_transfer_engines']['modtran']['lut_names'] = [x for x in ['H2OSTR','AERFRAC_2','GNDALT','H1ALT','TRUEAZ','OBSZEN', 'solzen','solaz'] if x in radiative_transfer_config['lut_grid'].keys()]
    radiative_transfer_config['radiative_transfer_engines']['sixs']['lut_names'] = [x for x in ['H2OSTR','AOT550','elev','alt','viewaz','viewzen','solzen','solaz'] if x in radiative_transfer_config['lut_grid'].keys()]

    # make isofit configuration
    isofit_config_modtran = {'input': {},
                             'output': {},
                             'forward_model': {
                                 "radiative_transfer": radiative_transfer_config,
                                 "instrument": {"wavelength_file": paths.wavelenth_file}
                             },
                             "implementation": {
                                "inversion": {"windows": [[1,2],[3,4]]},
                                "n_cores": n_cores}
                             }

    isofit_config_modtran['implementation']["rte_configure_and_exit"] = True

    # write modtran_template
    with open(config_output_path, 'w') as fout:
        fout.write(json.dumps(isofit_config_modtran, cls=SerialEncoder, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
