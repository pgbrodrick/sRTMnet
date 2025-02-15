






import numpy as np
import argparse
import h5py
import subprocess
import os
import ray
import multiprocessing as mp



def spectral_response_function(response_range: np.array, mu: float, sigma: float):
    """Calculate the spectral response function.

    Args:
        response_range: signal range to calculate over
        mu: mean signal value
        sigma: signal variation

    Returns:
        np.array: spectral response function

    """

    u = (response_range - mu) / abs(sigma)
    y = (1.0 / (np.sqrt(2.0 * np.pi) * abs(sigma))) * np.exp(-u * u / 2.0)
    srf = y / y.sum()
    return srf


def resample_spectrum(
    x: np.array, wl: np.array, wl2: np.array, fwhm2: np.array, fill: bool = False, H: np.array = None
) -> np.array:
    """Resample a spectrum to a new wavelength / FWHM.
       Assumes Gaussian SRFs.

    Args:
        x: radiance vector
        wl: sample starting wavelengths
        wl2: wavelengths to resample to
        fwhm2: full-width-half-max at resample resolution
        fill: boolean indicating whether to fill in extrapolated regions

    Returns:
        np.array: interpolated radiance vector

    """
    if H is None:
        H = np.array(
            [
                spectral_response_function(wl, wi, fwhmi / 2.355)
                for wi, fwhmi in zip(wl2, fwhm2)
            ]
        )
        H[np.isnan(H)] = 0

    dims = len(x.shape)
    if fill:
        if dims > 1:
            raise Exception("resample_spectrum(fill=True) only works with vectors")

        x = x.reshape(-1, 1)
        xnew = np.dot(H, x).ravel()
        good = np.isfinite(xnew)
        for i, xi in enumerate(xnew):
            if not good[i]:
                nearest_good_ind = np.argmin(abs(wl2[good] - wl2[i]))
                xnew[i] = xnew[nearest_good_ind]
        return xnew
    else:
        # Replace NaNs with zeros
        x[np.isnan(x)] = 0

        # Matrix
        if dims > 1:
            return np.dot(H, x.T).T

        # Vector
        else:
            x = x.reshape(-1, 1)
            return np.dot(H, x).ravel()


def main():
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--output_sixs_wl_file', type=str, default=None)
    parser.add_argument('--output_modtran_wl_file', type=str, default=None)
    parser.add_argument('--irr_file', type=str, default=None)
    args = parser.parse_args()

    output_sixs = None
    output_sixs_wl = None
    npzf = np.load(args.input_file)


    # SIXS
    input_sixs_wl = npzf['sixs_wavelengths']


    input_sixs = npzf['sixs_results'].copy()
    input_solar_irr = npzf['sol_irr']
    points = npzf['points']
    point_names = npzf['point_names'].tolist()

    if args.output_sixs_wl_file is not None:
        if args.irr_file is not None:
            irr_wl, irr = np.loadtxt(args.irr_file, comments="#").T
            irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
            input_sixs_irr = np.array(resample_spectrum(irr, irr_wl, input_sixs_wl, np.ones(len(irr_wl))*(input_sixs_wl[1]-input_sixs_wl[0])),dtype=np.float32)

        n_bands = int(input_sixs.shape[1] / len(npzf['keys']))
        for key in range(len(npzf['keys'])):
            input_sixs[:, n_bands * key : n_bands * (key + 1)] = input_sixs[:, n_bands * key : n_bands * (key + 1)] * input_sixs_irr * np.pi / np.cos(np.deg2rad(points[:, point_names.index('solar_zenith')]))[:,np.newaxis]
        
        output_sixs_wl = np.genfromtxt(args.output_sixs_wl_file)[:, 1]
        output_sixs_fwhm = np.genfromtxt(args.output_sixs_wl_file)[:, 2]
        if np.all(output_sixs_wl < 1000):
            output_sixs_wl *= 1000
            output_sixs_fwhm *= 1000

        output_sixs_irr = np.array(resample_spectrum(irr, irr_wl, output_sixs_wl, output_sixs_fwhm),dtype=np.float32)
        H = np.array( [ spectral_response_function(input_sixs_wl, wi, fwhmi) for wi, fwhmi in zip(output_sixs_wl, output_sixs_fwhm) ] )
        H[np.isnan(H)] = 0
        output_sixs = np.zeros((input_sixs.shape[0], len(output_sixs_wl)*len(npzf['keys'])))
        n_bands_o = len(output_sixs_wl)
        for key in range(len(npzf['keys'])):
            output_sixs[:, n_bands_o * key : n_bands_o * (key + 1)] = np.dot(input_sixs[:, n_bands * key : n_bands * (key + 1)], H.T) / output_sixs_irr / np.pi * np.cos(np.deg2rad(points[:, point_names.index('solar_zenith')]))[:,np.newaxis]
        print(output_sixs.shape)
    else:
        output_sixs = input_sixs
        output_sixs_wl = npzf['sixs_wavelengths']
    

    # MODTRAN
    output_modtran = None
    output_solar_irr = None
    output_modtran_wl = None
    input_modtran_wl = npzf['modtran_wavelengths']


    input_modtran = npzf['modtran_results'].copy()
    input_solar_irr = npzf['sol_irr']
    points = npzf['points']
    point_names = npzf['point_names'].tolist()

    if args.output_modtran_wl_file is not None:
        if args.irr_file is not None:
            irr_wl, irr = np.loadtxt(args.irr_file, comments="#").T
            irr = irr / 10
            input_modtran_irr = np.array(resample_spectrum(irr, irr_wl, input_modtran_wl, np.ones(len(irr_wl))*(input_modtran_wl[1] - input_modtran_wl[0])),dtype=np.float32)

        n_bands = int(input_modtran.shape[1] / len(npzf['keys']))
        #for key in range(len(npzf['keys'])):
        #    input_modtran[:, n_bands * key : n_bands * (key + 1)] = input_modtran[:, n_bands * key : n_bands * (key + 1)] * input_modtran_irr * np.pi / np.cos(np.deg2rad(points[:, point_names.index('solar_zenith')]))[:,np.newaxis]


        output_modtran_wl = np.genfromtxt(args.output_modtran_wl_file)[:, 1]
        output_modtran_fwhm = np.genfromtxt(args.output_modtran_wl_file)[:, 2]
        if np.all(output_modtran_wl < 1000):
            output_modtran_wl *= 1000
            output_modtran_fwhm *= 1000

        output_modtran_irr = np.array(resample_spectrum(irr, irr_wl, output_modtran_wl, output_modtran_fwhm),dtype=np.float32)
        H = np.array( [ spectral_response_function(input_modtran_wl, wi, fwhmi) for wi, fwhmi in zip(output_modtran_wl, output_modtran_fwhm) ] )
        H[np.isnan(H)] = 0
        output_modtran = np.zeros((input_modtran.shape[0], len(output_modtran_wl)*len(npzf['keys'])))
        n_bands_o = len(output_modtran_wl)
        input_modtran[np.isfinite(input_modtran) == False] = 0
        for key in range(len(npzf['keys'])):
            print(n_bands_o * key , n_bands_o * (key + 1), n_bands * key , n_bands * (key + 1))
            output_modtran[:, n_bands_o * key : n_bands_o * (key + 1)] = np.dot(H, input_modtran[:, n_bands * key : n_bands * (key + 1)].T).T #/ output_modtran_irr / np.pi * np.cos(np.deg2rad(points[:, point_names.index('solar_zenith')]))[:,np.newaxis]
            #import ipdb; ipdb.set_trace()
            #test=resample_spectrum(input_modtran[0, n_bands * key : n_bands * (key + 1)], input_modtran_wl, output_modtran_wl, output_modtran_fwhm)
            
        print(output_modtran.shape)


    else:
        output_modtran = input_modtran
        output_modtran_irr = npzf['sol_irr']
        output_modtran_wl = npzf['modtran_wavelengths']
    
    
    np.savez(args.output_file, modtran_results=output_modtran, 
                         sixs_results=output_sixs, 
                         points=npzf['points'],
                         sol_irr=output_modtran_irr,
                         sixs_wavelengths=output_sixs_wl,
                         modtran_wavelengths=output_modtran_wl,
                         point_names=npzf['point_names'],  
                         keys=npzf['keys']
                         )



        

if __name__ == "__main__":
    main()







