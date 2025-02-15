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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from isofit.radiative_transfer.engines.modtran import ModtranRT
from isofit.radiative_transfer.engines.six_s import SixSRT
from isofit.configs import configs
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import ray
from isofit.radiative_transfer.radiative_transfer import confPriority 



def d2_subset(data,ranges):
    a = data.copy()
    a = a[ranges[0],:]
    a = a[:,ranges[1]]
    return a

def readModtran(modtran_obj, filename):
    try:
        solzen = modtran_obj.load_tp6(f"{filename}.tp6")
        coszen = np.cos(solzen * np.pi / 180.0)
        params = modtran_obj.load_chn(f"{filename}.chn", coszen)

        # Remove thermal terms in VSWIR runs to avoid incorrect usage
        if modtran_obj.treat_as_emissive is False:
            for key in ["thermal_upwelling", "thermal_downwelling"]:
                if key in params:
                    Logger.debug(
                        f"Deleting key because treat_as_emissive is False: {key}"
                    )
                    del params[key]

        params["solzen"] = solzen
        params["coszen"] = coszen

        return params
    except:
        return None

def readSixs(sixs_obj, filename, wl, multipart_transmittance=False, wl_size=0):

    return sixs_obj.parse_file(
            file=file,
            wl=sixs_obj.wl,
            multipart_transmittance=sixs_obj.multipart_transmittance,
            wl_size=sixs_obj.wl.size,
        )

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('-config_file', type=str, default='templates/isofit_template.json')
    parser.add_argument('-keys', type=str, default=['rhoatm', 'sphalb', 'transm_down_dir', 'transm_down_dif', 'transm_up_dir', 'transm_up_dif' ], nargs='+')
    parser.add_argument('-munge_dir', type=str, default='munged')
    parser.add_argument('-figs_dir', type=str, default=None)
    parser.add_argument('-unstruct', type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    np.random.seed(13)




    config = configs.create_new_config(args.config_file)

    # Note - this goes way faster if you comment out the Vector Interpolater build section in each of these

    rt_config = config.forward_model.radiative_transfer
    instrument_config = config.forward_model.instrument

    
    params = {'engine_config': rt_config.radiative_transfer_engines[0]}
    
    params['lut_grid'] = confPriority('lut_grid', [params['engine_config'], instrument_config, rt_config])
    params['lut_grid'] = {key: params['lut_grid'][key] for key in params['engine_config'].lut_names.keys()}
    params['wavelength_file'] = confPriority('wavelength_file', [params['engine_config'], instrument_config, rt_config])
    if args.unstruct:
        params['engine_config'].rte_configure_and_exit = True
    else:
        params['engine_config'].rte_configure_and_exit = False
    #params['engine_config'].rt_mode = 'rdn'
    isofit_modtran = ModtranRT(**params)

    params = {'engine_config' : rt_config.radiative_transfer_engines[1]}
    
    params['lut_grid'] = confPriority('lut_grid', [params['engine_config'], instrument_config, rt_config])
    params['lut_grid'] = {key: params['lut_grid'][key] for key in params['engine_config'].lut_names.keys()}
    if args.unstruct:
        params['engine_config'].rte_configure_and_exit = True
    else:
        params['engine_config'].rte_configure_and_exit = False
    #params['engine_config'].rt_mode = 'rdn'

    # Get raw 6s return, not wavelength convolved (this is what we'll use for inference too)
    sixs_wl = np.arange(350, 2500 + 2.5, 2.5)
    sixs_fwhm = np.full(sixs_wl.size, 2.0)
    #params['wavelength_file'] = confPriority('wavelength_file', [params['engine_config'], instrument_config, rt_config])
    params['wl'] = sixs_wl
    params['fwhm'] = sixs_fwhm
    isofit_sixs = SixSRT(**params)






    outdict_sixs, outdict_modtran = {}, {}
    munge_file = os.path.join(args.munge_dir, 'combined_data.npz')
    good_points = np.ones(isofit_sixs.lut['rhoatm'].shape[0]).astype(bool)
    for key_ind, key in enumerate(args.keys):

            if args.figs_dir is not None:
                point_dims = list(isofit_modtran.lut_grid.keys())
                point_dims_s = list(isofit_sixs.lut_grid.keys())
                rtm_key = 'transm_down_dir'
                for _ind in range(isofit_sixs.lut['rhoatm'].shape[0]):

                    sp = isofit_sixs.points[_ind,:]
                    mp = isofit_modtran.points[_ind,:]

                    std_dir = isofit_sixs.lut['transm_down_dir'][_ind,:]
                    mtd_dir = isofit_modtran.lut['transm_down_dir'][_ind,:]
                    std_dif = isofit_sixs.lut['transm_down_dif'][_ind,:]
                    mtd_dif = isofit_modtran.lut['transm_down_dif'][_ind,:]
                    stu_dir = isofit_sixs.lut['transm_up_dir'][_ind,:]
                    mtu_dir = isofit_modtran.lut['transm_up_dir'][_ind,:]
                    stu_dif = isofit_sixs.lut['transm_up_dif'][_ind,:]
                    mtu_dif = isofit_modtran.lut['transm_up_dif'][_ind,:]
                    s_r = isofit_sixs.lut['rhoatm'][_ind,:]
                    m_r = isofit_modtran.lut['rhoatm'][_ind,:]

                    name='_'.join([f'{point_dims_s[x]}_{sp[x]}' for x in range(len(sp))])
                    plt.plot(isofit_sixs.wl, std_dir, color='black', label='t_down_dir')
                    plt.plot(isofit_sixs.wl, std_dif, color='grey', label='t_down_dif')
                    plt.plot(isofit_sixs.wl, stu_dir, color='red', label='t_up_dir')
                    plt.plot(isofit_sixs.wl, stu_dif, color='purple', label='t_up_dif')
                    plt.plot(isofit_sixs.wl, s_r, color='green', label='rhoatm')
                    
                    name2='_'.join([f'{point_dims[x]}_{mp[x]}' for x in range(len(mp))])
                    plt.plot(isofit_modtran.wl, mtd_dir, color='black', ls = '--')
                    plt.plot(isofit_modtran.wl, mtd_dif, color='grey', ls = '--')
                    plt.plot(isofit_modtran.wl, mtu_dir, color='red', ls = '--')
                    plt.plot(isofit_modtran.wl, mtu_dif, color='purple', ls = '--')
                    plt.plot(isofit_modtran.wl, m_r, color='green', ls = '--')

                    plt.legend(fontsize=4, loc='lower right')
                    
                    plt.title(f'S: {name}\n M: {name2}',fontsize=6)

                    plt.savefig(f'{args.figs_dir}/{name.replace("\n","_")}.png',dpi=100)
                    plt.clf()

            point_names = isofit_sixs.lut.point.to_index().names
            bad_points = np.zeros(isofit_sixs.lut[key].shape[0],dtype=bool)
            if 'surface_elevation_km' in point_names and 'observer_altitude_km' in point_names:
                bad_points = isofit_sixs.lut['surface_elevation_km']  >= isofit_sixs.lut['observer_altitude_km'] -2
                bad_points[np.any(isofit_sixs.lut['transm_down_dif'].data[:,:] > 10,axis=1)] = True
                bad_points[np.any(isofit_modtran.lut['transm_down_dif'].data[:,:] > 10,axis=1)] = True
                good_points = np.logical_not(bad_points)
            
            outdict_sixs[key] = np.array(isofit_sixs.lut[key])[good_points,:]
            outdict_modtran[key] = np.array(isofit_modtran.lut[key])[good_points,:]


    # hack at some things to prevent runaway values
    outdict_sixs['transm_up_dif'][outdict_sixs['transm_up_dif'][:,:] > 1] = 0
    outdict_sixs['transm_up_dif'][outdict_sixs['transm_up_dif'][:,:] < 0] = 0
    outdict_modtran['transm_up_dif'][outdict_modtran['transm_up_dif'][:,:] > 1] = 0
    outdict_modtran['transm_up_dif'][outdict_modtran['transm_up_dif'][:,:] < 0] = 0

    outdict_sixs['sphalb'][:,isofit_sixs.wl > 1200][outdict_sixs['sphalb'][:,isofit_sixs.wl > 1200] > 0.1] = 0
    outdict_sixs['sphalb'][outdict_sixs['sphalb'][:,:] < 0] = 0

    subs = outdict_modtran['sphalb'][:,isofit_modtran.wl > 1200].copy()
    subs[subs > 0.1] = 0
    outdict_modtran['sphalb'][:,isofit_modtran.wl > 1200] = subs
    print(np.sum(outdict_modtran['sphalb'][:,isofit_modtran.wl > 1200] > 0.1))

    outdict_modtran['sphalb'][outdict_modtran['sphalb'][:,:] < 0] = 0



    #keys=list(isofit_modtran.lut_grid.keys())
    keys=['rhoatm','sphalb','transm_down_dir','transm_down_dif', 'transm_up_dir','transm_up_dif']
    #keys=list(isofit_sixs.lut.point.to_index().names)
    stacked_modtran = np.zeros((outdict_modtran[keys[0]].shape[0], outdict_modtran[keys[0]].shape[1] * len(keys)))
    stacked_sixs = np.zeros((outdict_sixs[keys[0]].shape[0], outdict_sixs[keys[0]].shape[1] * len(keys)))
    n_bands_modtran = int(stacked_modtran.shape[-1]/len(keys))
    n_bands_sixs = int(stacked_sixs.shape[-1]/len(keys))
    for n in range(len(keys)):
        stacked_modtran[:,n*n_bands_modtran:(n+1)*n_bands_modtran] = outdict_modtran[keys[n]]
        stacked_sixs[:,n*n_bands_sixs:(n+1)*n_bands_sixs] = outdict_sixs[keys[n]]


    np.savez(munge_file, modtran_results=stacked_modtran, 
                         sixs_results=stacked_sixs, 
                         points=isofit_sixs.points[good_points,:],
                         sol_irr=isofit_modtran.lut.solar_irr,
                         sixs_wavelengths=isofit_sixs.wl,
                         modtran_wavelengths=isofit_modtran.wl,
                         point_names=list(isofit_sixs.lut.point.to_index().names),
                         keys=keys
                         )



if __name__ == '__main__':
    main()
