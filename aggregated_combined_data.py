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
from isofit.radiative_transfer.modtran import ModtranRT
from isofit.radiative_transfer.six_s import SixSRT
from isofit.configs import configs
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import ray



def d2_subset(data,ranges):
    a = data.copy()
    a = a[ranges[0],:]
    a = a[:,ranges[1]]
    return a


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('-config_file', type=str, default='templates/isofit_template.json')
    parser.add_argument('-keys', type=str, default=['transm', 'rhoatm', 'sphalb'], nargs='+')
    parser.add_argument('-munge_dir', type=str, default='munged')

    args = parser.parse_args()

    np.random.seed(13)

    for key_ind, key in enumerate(args.keys):
        munge_file = os.path.join(args.munge_dir, key + '.npz')

        if os.path.isfile(munge_file) is False:
            config = configs.create_new_config(args.config_file)

            # Note - this goes way faster if you comment out the Vector Interpolater build section in each of these
            isofit_modtran = ModtranRT(config.forward_model.radiative_transfer.radiative_transfer_engines[0],
                                       config, build_lut = False)
            isofit_sixs = SixSRT(config.forward_model.radiative_transfer.radiative_transfer_engines[1],
                                 config, build_lut = False)

            sixs_results = get_obj_res(isofit_sixs, key, resample=False)
            modtran_results = get_obj_res(isofit_modtran, key)

            if os.path.isdir(os.path.dirname(munge_file) is False):
                os.mkdir(os.path.dirname(munge_file))

            for fn in isofit_modtran.files:
                mod_output = isofit_modtran.load_rt(fn)
                sol_irr = mod_output['sol']
                if np.all(np.isfinite(sol_irr)):
                    break

            np.savez(munge_file, modtran_results=modtran_results, sixs_results=sixs_results, sol_irr=sol_irr)

    modtran_results = None
    sixs_results = None
    for key_ind, key in enumerate(args.keys):
        munge_file = os.path.join(args.munge_dir, key + '.npz')

        npzf = np.load(munge_file)

        dim1 = int(np.product(np.array(npzf['modtran_results'].shape)[:-1]))
        dim2 = npzf['modtran_results'].shape[-1]
        if modtran_results is None:
            modtran_results = np.zeros((dim1,dim2*len(args.keys)))
        modtran_results[:,dim2*key_ind:dim2*(key_ind+1)] = npzf['modtran_results']

        dim1 = int(np.product(np.array(npzf['sixs_results'].shape)[:-1]))
        dim2 = npzf['sixs_results'].shape[-1]
        if sixs_results is None:
            sixs_results = np.zeros((dim1,dim2*len(args.keys)))
        sixs_results[:,dim2*key_ind:dim2*(key_ind+1)] = npzf['sixs_results']

        sol_irr = npzf['sol_irr']


    config = configs.create_new_config(args.config_file)
    isofit_modtran = ModtranRT(config.forward_model.radiative_transfer.radiative_transfer_engines[0],
                               config, build_lut=False)
    isofit_sixs = SixSRT(config.forward_model.radiative_transfer.radiative_transfer_engines[1],
                        config, build_lut=False)
    sixs_names = isofit_sixs.lut_names
    modtran_names = isofit_modtran.lut_names

    if 'elev' in sixs_names:
        sixs_names[sixs_names.index('elev')] = 'GNDALT'
    if 'viewzen' in sixs_names:
        sixs_names[sixs_names.index('viewzen')] = 'OBSZEN'
    if 'viewaz' in sixs_names:
        sixs_names[sixs_names.index('viewaz')] = 'TRUEAZ'
    if 'alt' in sixs_names:
        sixs_names[sixs_names.index('alt')] = 'H1ALT'
    if 'AOT550' in sixs_names:
        sixs_names[sixs_names.index('AOT550')] = 'AERFRAC_2'

    reorder_sixs = [sixs_names.index(x) for x in modtran_names]

    points = isofit_modtran.points.copy()
    points_sixs = isofit_sixs.points.copy()[:,reorder_sixs]

    if 'OBSZEN' in modtran_names:
        print('adjusting')
        points_sixs[:, modtran_names.index('OBSZEN')] = 180 - points_sixs[:, modtran_names.index('OBSZEN')]


    ind = np.lexsort(tuple([points[:,x] for x in range(points.shape[-1])]))
    points = points[ind,:]
    modtran_results = modtran_results[ind,:]

    ind_sixs = np.lexsort(tuple([points_sixs[:,x] for x in range(points_sixs.shape[-1])]))
    points_sixs = points_sixs[ind_sixs,:]
    sixs_results = sixs_results[ind_sixs,:]


    good_data = np.all(np.isnan(modtran_results) == False,axis=1)
    good_data[np.any(np.isnan(sixs_results),axis=1)] = False

    modtran_results = modtran_results[good_data,:]
    sixs_results = sixs_results[good_data,:]
    points = points[good_data,...]
    points_sixs = points_sixs[good_data,...]

    print(sixs_results.shape)
    print(modtran_results.shape)

    tmp = isofit_sixs.load_rt(isofit_sixs.files[0])

    np.savez(os.path.join(args.munge_dir, 'combined_training_data.npz'), sixs_results=sixs_results, modtran_results=modtran_results,
             points=points, points_sixs=points_sixs, keys=args.keys, point_names=modtran_names, modtran_wavelengths=isofit_modtran.wl,
             sixs_wavelengths=isofit_sixs.grid,
             sol_irr=sol_irr)


@ray.remote
def read_data_piece(ind, maxind, point, fn, key, resample, obj):
    if ind % 100 == 0:
        print('{}: {}/{}'.format(key, ind, maxind))
    try:
        if resample is False:
            mod_output = obj.load_rt(fn, resample=False)
        else:
            mod_output = obj.load_rt(fn)
        res = mod_output[key]
    except:
        res = None
    return ind, res



def get_obj_res(obj, key, resample=True):

    # We don't want the VectorInterpolator, but rather the raw inputs
    ray.init()
    if hasattr(obj,'sixs_ngrid_init'):
        results = np.zeros((obj.points.shape[0],obj.sixs_ngrid_init), dtype=float)
    else:
        results = np.zeros((obj.points.shape[0],obj.n_chan), dtype=float)
    objid = ray.put(obj)
    jobs = []
    for ind, (point, fn) in enumerate(zip(obj.points, obj.files)):
        jobs.append(read_data_piece.remote(ind, results.shape[0], point, fn, key, resample, objid))
    rreturn = [ray.get(jid) for jid in jobs]
    for ind, res in rreturn:
        if res is not None:
            try:
                results[ind,:] = res
            except:
                results[ind,:] = np.nan
        else:
            results[ind,:] = np.nan
    ray.shutdown()
    return results


if __name__ == '__main__':
    main()
