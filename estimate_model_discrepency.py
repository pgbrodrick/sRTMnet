


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from isofit.radiative_transfer.modtran import ModtranRT
from isofit.radiative_transfer.six_s import SixSRT
from isofit.configs import configs
import argparse
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
#from sklearn.externals import joblib 
from isofit.core.common import resample_spectrum
import scipy.io
from scipy import interpolate
from isofit.core.common import resample_spectrum
import ray


def get_model_results(save_dir, dim, slice, points, test=True):

    base_save_name = os.path.join(save_dir,'emulator')
    if test:
        base_save_name += 'dim_{}_slice_{}'.format(dim, slice)
    else:
        #base_save_name += 'dim_-2_slice_0'.format(dim, slice)
        base_save_name += '_random'.format(dim, slice)

    model_file = base_save_name
    model_aux_file = base_save_name + '_aux.npz'
    results_file = base_save_name + '_pred_results.npz'

    results_npzf = np.load(results_file)
    results = results_npzf['predicted_modtran']
    
    un_dim = np.unique(points[:,dim])

    test = np.where(points[:,dim] == un_dim[slice])[0]
    train = np.where(points[:,dim] != un_dim[slice])[0]

    return results[test,:], test
    #return results, test

@ray.remote
def forward_model(index, coefficients, rfl, coszen, solar_irr):

    n_bands = int(len(coefficients)/3)

    transm = coefficients[:n_bands]
    rho = coefficients[n_bands:2*n_bands]
    sphalb = coefficients[n_bands*2:n_bands*3]

    rdn_atm = rho / np.pi*(solar_irr * coszen)

    rdn_down = (solar_irr * coszen) / np.pi * transm

    rdn = rdn_atm + rdn_down * rfl / (1.0 - sphalb * rfl)

    return index, rdn
 


@ray.remote
def resample_single(ind, ind_emulator_output, emulator_wavelengths, wavelengths, fwhm): 
    if ind %1000  == 0:
        print(ind)
    return ind, resample_spectrum(ind_emulator_output, emulator_wavelengths, wavelengths, fwhm)

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('munged_file', type=str, default='munged/combined_training_data.npz')
    parser.add_argument('model_file', type=str)
    parser.add_argument('aux_file',type=str,default=None)
    parser.add_argument('wavelength_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('-fig_dir', type=str, default='figs')
    args = parser.parse_args()
    np.random.seed(13)

    npzf = np.load(args.munged_file)

    if args.aux_file is None:
        args.aux_file = args.model_file + '_aux.npz'
    model_aux = np.load(args.aux_file)

    modtran_results = npzf['modtran_results']
    sixs_results = npzf['sixs_results']
    points = npzf['points']

    keys = npzf['keys']
    point_names = npzf['point_names'].tolist()
    simulator_wavelengths = npzf['sixs_wavelengths']
    emulator_wavelengths = npzf['modtran_wavelengths']
    solar_irr = npzf['sol_irr']
    n_bands_modtran = int(modtran_results.shape[-1]/len(keys))
    n_bands_sixs = int(sixs_results.shape[-1]/len(keys))

    sixs_results_match_modtran = np.zeros(modtran_results.shape)
    for key_ind, key in enumerate(keys):
        band_range_m = np.arange(n_bands_modtran * key_ind, n_bands_modtran * (key_ind + 1))
        band_range_s = np.arange(n_bands_sixs * key_ind, n_bands_sixs * (key_ind + 1))

        x = simulator_wavelengths
        y = sixs_results[:,band_range_s]
        print(x.shape, y.shape)
        finterp = interpolate.interp1d(x,y)
        sixs_results_match_modtran[:,band_range_m] = finterp(emulator_wavelengths)

    if 'response_scaler' in model_aux.keys():
        response_scaler = model_aux['response_scaler']
    else:  
        response_scaler = 100
    print(response_scaler)
    model = keras.models.load_model(args.model_file) 

    full_pred = model.predict(sixs_results)/response_scaler
    full_pred += sixs_results_match_modtran

    all_wl_info = np.genfromtxt(args.wavelength_file)
    wl = all_wl_info[:,1]
    fwhm = all_wl_info[:,2]
    if np.all(wl < 100):
        wl *= 1000
        fwhm *= 1000


    ray.init()
    modtran_coeff_interp = np.zeros((modtran_results.shape[0],len(wl)*len(keys)))-9999
    for key_ind, key in enumerate(keys):
        resample_jobs = []
        for point_ind in range(sixs_results.shape[0]):
            leo = modtran_results[point_ind, key_ind*len(emulator_wavelengths): (key_ind+1)*len(emulator_wavelengths)]
            resample_jobs.append(resample_single.remote(point_ind,leo,emulator_wavelengths,wl,fwhm))
        resample_results = ray.get(resample_jobs)
        
        for ind, res in resample_results:
            modtran_coeff_interp[ind, key_ind*len(wl): (key_ind+1)*len(wl)] = res

    emu_coeff_interp = np.zeros((modtran_results.shape[0],len(wl)*len(keys)))-9999
    for key_ind, key in enumerate(keys):
        resample_jobs = []
        for point_ind in range(sixs_results.shape[0]):
            leo = full_pred[point_ind, key_ind*len(emulator_wavelengths): (key_ind+1)*len(emulator_wavelengths)]
            resample_jobs.append(resample_single.remote(point_ind,leo,emulator_wavelengths,wl,fwhm))
        resample_results = ray.get(resample_jobs)
        
        for ind, res in resample_results:
            emu_coeff_interp[ind, key_ind*len(wl): (key_ind+1)*len(wl)] = res

    
    resample_irr = resample_spectrum(solar_irr, emulator_wavelengths, wl, fwhm)
    print(resample_irr.shape)
    reference_refl = np.ones(len(wl)) * 0.5
    
    emulator_radiance = np.zeros((emu_coeff_interp.shape[0], len(wl)))
    print(emulator_radiance.shape)
    print(emu_coeff_interp[point_ind,:].shape)
    print(modtran_coeff_interp[point_ind,:].shape)
    resample_jobs = []
    for point_ind in range(emu_coeff_interp.shape[0]):
        resample_jobs.append(forward_model.remote(point_ind, emu_coeff_interp[point_ind,:],reference_refl, np.cos(np.deg2rad(points[point_ind, point_names.index('solzen')])), resample_irr))
    resample_results = ray.get(resample_jobs)
    for ind, res in resample_results:
        emulator_radiance[ind, :] = res
        
    modtran_radiance = np.zeros((emu_coeff_interp.shape[0], len(wl)))
    resample_jobs = []
    for point_ind in range(emu_coeff_interp.shape[0]):
        resample_jobs.append(forward_model.remote(point_ind, modtran_coeff_interp[point_ind,:],reference_refl, np.cos(np.deg2rad(points[point_ind, point_names.index('solzen')])), resample_irr))
    resample_results = ray.get(resample_jobs)
    for ind, res in resample_results:
        modtran_radiance[ind, :] = res

    delta = np.mean(np.power(modtran_radiance - emulator_radiance,2), axis=0)
    cov_output = np.zeros((len(delta),len(delta)))
    for n in range(len(delta)):
        cov_output[n,n] = delta[n]
   
    np.savez(os.path.splitext(args.output_file)[0] + '_delta_rad.npz', delta=modtran_radiance-emulator_radiance, points=points)
    scipy.io.savemat(args.output_file, {'cov': cov_output})

            



if __name__ == '__main__':
    main()






    

