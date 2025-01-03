



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#from isofit.radiative_transfer.modtran import ModtranRT
#from isofit.radiative_transfer.six_s import SixSRT
from isofit.configs import configs
import argparse
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import tensorflow
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
#from sklearn.externals import joblib 
from isofit.core.common import resample_spectrum
from scipy import interpolate
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def rho_to_rdn(rho,solar_irr,coszen):
    return rho / np.pi * solar_irr[np.newaxis,:] * coszen



def beckman_rdn(simulated_modtran, wl, n_bands=424):


    refl_file = "/store/brodrick/repos/isofit-tutorials/20171108_Pasadena/insitu/BeckmanLawn.txt"
    solar_irr = np.genfromtxt('/store/brodrick/repos/isofit-data/kurudz_0.1nm.dat')[:-1]


    coszen = 0.6155647578988601
    rfl = np.genfromtxt(refl_file)
    fwhm = rfl[:,2]*1000
    print(fwhm[0])
    rfl = resample_spectrum(rfl[:,1],rfl[:,0],wl,fwhm)

    irr = np.loadtxt('/store/brodrick/repos/isofit-data/kurudz_0.1nm.dat', comments="#")
    iwl, irr = irr.T
    irr = irr / 10.0  # convert, uW/nm/cm2
    #irr = irr / self.irr_factor**2  # consider solar distance
    solar_irr = resample_spectrum(irr, iwl, wl, fwhm)

    n_bands = len(wl)

    #keys=['rhoatm','sphalb','transm_down_dir','transm_down_dif', 'transm_up_dir','transm_up_dif'] 
    rho = simulated_modtran[:,:n_bands]
    sphalb = simulated_modtran[:,n_bands*1:n_bands*2]
    transm_down_dir = simulated_modtran[:,n_bands*2:n_bands*3]
    transm_down_dif = simulated_modtran[:,n_bands*3:n_bands*4]
    transm_up_dir = simulated_modtran[:,n_bands*4:n_bands*5]
    transm_up_dif = simulated_modtran[:,n_bands*5:n_bands*6]


    # bi-directional            (downward direct * upward direct)
    # hemispherical-directional (downward diffuse * upward direct)
    # directional-hemispherical (downward direct * upward diffuse)
    # bi-hemispherical          (downward diffuse * upward diffuse)
    ## at-sensor radiance model, including topography, adjacency effects, and glint
    #ret = (
    #    L_atm
    #    + (
    #        L_bi_direct * rfl_dir  # bi-directional radiance
    #        + L_hemi_direct * rfl_dif  # hemispherical-directional radiance
    #        + L_direct_hemi * bg_dir  # directional-hemispherical radiance
    #        + L_bi_hemi * bg_dif  # bi-hemispherical radiance
    #    )
    #    / (1.0 - s_alb * bg_dif)
    #    + L_up
    #)


    L_bi_direct = rho_to_rdn(transm_down_dir * transm_up_dir, solar_irr, coszen)
    L_hemi_direct = rho_to_rdn(transm_down_dif * transm_up_dir, solar_irr, coszen)
    L_direct_hemi = rho_to_rdn(transm_down_dir * transm_up_dif, solar_irr, coszen)
    L_bi_hemi = rho_to_rdn(transm_down_dif * transm_up_dif, solar_irr, coszen)
    L_atm = rho_to_rdn(rho, solar_irr, coszen)

    rdn =  L_atm +\
        (+\
        L_bi_direct * rfl +\
        L_hemi_direct * rfl +\
        L_direct_hemi * rfl +\
        L_bi_hemi * rfl\
        ) / (1.0 - sphalb * rfl)


    rdn_atm = rho_to_rdn(rho, solar_irr, coszen)


    return rdn, rdn_atm
    


def nn_model(in_data_shape, out_data_shape, num_layers=5):

    layer_depths = np.linspace(in_data_shape[-1],out_data_shape[-1],num=num_layers+1,dtype=int)

    inlayer = keras.layers.Input(shape=(in_data_shape[-1],))
    output_layer = inlayer

    for _l in range(num_layers):
        output_layer = keras.layers.Dense(units=layer_depths[_l])(output_layer)
        output_layer = keras.layers.LeakyReLU(alpha=0.4)(output_layer)

    output_layer = keras.layers.Dense(units=out_data_shape[-1], activation='linear')(output_layer)
    model = keras.models.Model(inputs=[inlayer], outputs=[output_layer])
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    #optimizer=keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=optimizer)

    return model



def nn_model_ind(in_data_shape, out_data_shape, num_keys, num_layers=1):

    layer_depths = np.linspace(int(in_data_shape[-1]/num_keys),int(out_data_shape[-1]/num_keys),num=num_layers+1,dtype=int)

    inlayer = keras.layers.Input(shape=(in_data_shape[-1],))
    
    instack = []
    outstack = []
    input_width = int(in_data_shape[-1]/num_keys)
    for nk in range(num_keys):
        #output_layer = keras.layers.Cropping1D(cropping=(0,(num_keys-1)*(in_data_shape[-1]/num_keys)))(inlayer)
        output_layer = inlayer[:, nk*input_width:(nk+1)*input_width]

        for _l in range(num_layers):
            output_layer = keras.layers.Dense(units=layer_depths[_l])(output_layer)
            output_layer = keras.layers.LeakyReLU(alpha=0.4)(output_layer)

        output_layer = keras.layers.Dense(units=int(out_data_shape[-1]/num_keys), activation='linear')(output_layer)
        outstack.append(output_layer)

    output_merge = keras.layers.Concatenate()(outstack)

    model = keras.models.Model(inputs=[inlayer], outputs=[output_merge])
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
    #optimizer=keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=optimizer)

    return model

class SplitModel():

    def __init__(self,in_data_shape, out_data_shape, n_splits):
        self.in_data_shape = in_data_shape
        self.out_data_shape = out_data_shape
        self.n_splits = n_splits

        self.out_bandsets = np.linspace(0,out_data_shape[-1]-1,self.n_splits+1,dtype=int)

        self.individual_models = []
        for _m in range(self.n_splits):
            #lm = nn_model(in_data_shape, (1,self.out_bandsets[_m+1] - self.out_bandsets[_m]))
            lm = linear_model.LinearRegression()
            self.individual_models.append(lm)

    def fit(self, X, Y, validation_data):
        for _m, model in enumerate(self.individual_models):
            print('Training Model {}/{}'.format(_m,len(self.individual_models)))
            #model.fit(X,Y[:,self.out_bandsets[_m]:self.out_bandsets[_m+1]], epochs=25, batch_size=200, 
            #          validation_data=(validation_data[0],validation_data[1][:,self.out_bandsets[_m]:self.out_bandsets[_m+1]]))

            model.fit(X,Y[:,self.out_bandsets[_m]:self.out_bandsets[_m+1]])
    
    def predict(self, X):
        output = np.zeros((X.shape[0], self.out_data_shape[-1]))
        for _m, model in enumerate(self.individual_models):
            output[:,self.out_bandsets[_m]:self.out_bandsets[_m+1]] = model.predict(X)
        return output


def d2_subset(data,ranges):
    a = data.copy()
    a = a[ranges[0],:]
    a = a[:,ranges[1]]
    return a


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('munged_file', type=str, default='munged/combined_training_data.npz')
    parser.add_argument('-plot_figs', type=int, default=1)
    parser.add_argument('-fig_dir', type=str, default='figs')
    parser.add_argument('-save_dir', type=str, default='trained_models')
    parser.add_argument('-holdout_dim', type=int, default=-1)
    parser.add_argument('-holdout_slice', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(13)

    #es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
    #train_sixs = np.ones((13905,5166))
    #train_modtran = np.ones((13905, 129006))
    #keys = np.ones(6)
    #model = nn_model_ind(train_sixs.shape, train_modtran.shape, len(keys))
    #print(model.summary())
    #model.fit(train_sixs, train_modtran, batch_size=10, epochs=400,
    #          validation_data=(train_sixs, train_modtran),callbacks=[es])
    #exit()



    npzf = np.load(args.munged_file, allow_pickle=True)

    sixs_results = npzf['sixs_results']
    modtran_results = npzf['modtran_results']
    points = npzf['points']
    keys = npzf['keys']
    point_names = npzf['point_names']
    print(points.shape)
    simulator_wavelengths = npzf['sixs_wavelengths']
    emulator_wavelengths = npzf['modtran_wavelengths']
    solar_irr = npzf['sol_irr']
    print(keys)
    print(point_names)
    n_bands_modtran = int(modtran_results.shape[-1]/len(keys))
    n_bands_sixs = int(sixs_results.shape[-1]/len(keys))

    modtran_results[np.isnan(modtran_results)] = 0
    sixs_results[np.isnan(sixs_results)] = 0
    modtran_results[np.isfinite(modtran_results) == False] = 0
    sixs_results[np.isfinite(sixs_results) == False] = 0

    max_wl = np.min([np.max(simulator_wavelengths),np.max(emulator_wavelengths)])
    min_wl = np.max([np.min(simulator_wavelengths),np.min(emulator_wavelengths)])
    emulator_idx = np.where(np.logical_and(emulator_wavelengths >= min_wl, emulator_wavelengths <= max_wl))[0]

    modtran_results_clipped = np.zeros((modtran_results.shape[0], len(emulator_idx)*len(keys)))
    for n in range(len(keys)):
        modtran_results_clipped[:,n*len(emulator_idx):(n+1)*len(emulator_idx)] = modtran_results[:, emulator_idx + n_bands_modtran*n]

    print(modtran_results.shape)
    modtran_results = modtran_results_clipped
    del modtran_results_clipped
    n_bands_modtran = int(modtran_results.shape[-1]/len(keys))
    emulator_wavelengths = emulator_wavelengths[emulator_idx]

    print(modtran_results.shape)

    #fig = plt.figure(figsize=(20,5))
    #gs = gridspec.GridSpec(ncols=1, nrows=2, wspace=0.3, hspace=0.4)
    #ax = fig.add_subplot(gs[0, 0])
    #plt.plot(np.mean(modtran_results,axis=0))
    #ax = fig.add_subplot(gs[1, 0])
    #plt.plot(np.mean(sixs_results,axis=0))
    #plt.savefig('figs/h2o_comp/inputs.png',dpi=200,bbox_inches='tight')
    #exit()



    if args.holdout_dim == -1:
        perm = np.random.permutation(points.shape[0])
        test = perm[:int(0.1*len(perm))]
        train = perm[int(0.1*len(perm)):]
    elif args.holdout_dim == -2:
        perm = np.random.permutation(points.shape[0])
        train = perm.copy()
        test = train
        #test = perm[:int(0.1*len(perm))]
    else:
        if args.holdout_dim >= points.shape[1]:
            print('Holdout dim {} exceeds point dimension {}'.format(args.holdout_dim, points.shape[1]))
            quit()
        un_dim = np.unique(points[:,args.holdout_dim])
        if (args.holdout_slice >= len(un_dim)):
            print('Holdout slice {} > dim {} length: {}'.format(args.holdout_slice, args.holdout_dim, len(un_dim)))
            quit()
        test = np.where(points[:,args.holdout_dim] == un_dim[args.holdout_slice])[0]
        train = np.where(points[:,args.holdout_dim] != un_dim[args.holdout_slice])[0]


    print(test)
    print(train)

    band_range = np.arange(sixs_results.shape[-1])
    train_sixs = sixs_results[:,band_range]

    sixs_results_match_modtran = np.zeros(modtran_results.shape)
    for key_ind, key in enumerate(keys):
        band_range_m = np.arange(n_bands_modtran * key_ind, n_bands_modtran * (key_ind + 1))
        band_range_s = np.arange(n_bands_sixs * key_ind, n_bands_sixs * (key_ind + 1))

        x = simulator_wavelengths
        y = sixs_results[:,band_range_s]
        finterp = interpolate.interp1d(x,y)
        sixs_results_match_modtran[:,band_range_m] = finterp(emulator_wavelengths)


    train_modtran = modtran_results-sixs_results_match_modtran
    train_modtran[np.isfinite(train_modtran) == False] = 0
    train_sixs[np.isfinite(train_sixs) == False] = 0

    print(train_modtran.shape)
    base_save_name = os.path.join(args.save_dir,'emulator')
    if args.holdout_dim == -1:
        base_save_name += '_random'
    elif args.holdout_dim == -1:
        base_save_name += '_full'
    else:
        base_save_name += 'dim_{}_slice_{}'.format(args.holdout_dim, args.holdout_slice)

    monitor='val_loss'
        
    es = keras.callbacks.EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=20, restore_best_weights=True)
    #model = nn_model_ind(train_sixs.shape, train_modtran.shape, len(keys))
    model = nn_model(train_sixs.shape, train_modtran.shape)
    print(model.summary())


    simple_response_scaler = np.ones(train_modtran.shape[1])*100
    train_modtran *= simple_response_scaler
    #import ipdb; ipdb.set_trace()
    model.fit(train_sixs[train,:], train_modtran[train,:], batch_size=1000, epochs=400,
              validation_data=(train_sixs[test,:], train_modtran[test,:]),callbacks=[es])
    train_modtran /= simple_response_scaler

    full_pred = model.predict(train_sixs)/simple_response_scaler
    full_pred = full_pred + sixs_results_match_modtran
   
    pred = full_pred[test, :]

    model.save(base_save_name + '.h5')
    np.savez(base_save_name + '_aux.npz', lut_names=point_names, 
             rt_quantities=keys, 
             solar_irr=solar_irr, 
             emulator_wavelengths=emulator_wavelengths,
             response_scaler=simple_response_scaler,
             simulator_wavelengths=simulator_wavelengths)

    np.savez(base_save_name + '_pred_results.npz', predicted_modtran=full_pred)

    if args.plot_figs == 0:
        quit()

    rdn, rdn_atm = beckman_rdn(full_pred,emulator_wavelengths)
    print(rdn.shape)


    

    fig = plt.figure(figsize=(20, 20 * 2 / (0.5 + 2)))
    gs = gridspec.GridSpec(ncols=len(keys), nrows=2, wspace=0.3, hspace=0.4)

    ref_rdn = np.genfromtxt('/store/brodrick/repos/isofit-tutorials/20171108_Pasadena/remote/ang20171108t184227_rdn_v2p11_BeckmanLawn.txt')
    rdn_modtran, rdn_modtran_atm = beckman_rdn(modtran_results,emulator_wavelengths)

    ref_rdn = resample_spectrum(ref_rdn[:,1],ref_rdn[:,0],emulator_wavelengths, np.ones(len(emulator_wavelengths))*7.5)



    cf = 100

    varset = np.where(np.logical_and.reduce((points[:,0]  == np.median(points[:,0]),  
                                             points[:,1] == np.median(points[:,1]), 
                                             points[:,2] == np.median(points[:,2]), 
                                             points[:,3] == np.median(points[:,3]), 
                                             points[:,4] == np.median(points[:,4]), 
                                             points[:,5] == np.median(points[:,5]), 
                                             points[:,6] == np.median(points[:,6]), 
                                             )))[0]

    best_modtran = np.argmin(np.sum(np.power(rdn_modtran[:,:cf] - ref_rdn[:cf],2),axis=1))
    best_emu = np.argmin(np.sum(np.power(rdn[:,:cf] - ref_rdn[:cf],2),axis=1))
    plt.plot(emulator_wavelengths, ref_rdn, c='black', linewidth=0.8)

    #plt.fill_between(simulated_wavelengths, np.min(rdn_modtran[varset,:],axis=0), np.max(rdn_modtran[varset,:],axis=0),facecolor='red',alpha=0.5)
    #plt.fill_between(simulated_wavelengths, np.min(rdn[varset,:],axis=0), np.max(rdn[varset,:],axis=0),facecolor='green',alpha=0.5)

    plt.plot(emulator_wavelengths, rdn_modtran[varset,:].flatten(), c='red', linewidth=0.8, ls='--')
    plt.plot(emulator_wavelengths, rdn[varset,:].flatten(), c='green', linewidth=0.8, ls='--')
    
    plt.plot(emulator_wavelengths, rdn_modtran[best_modtran,:], c='red', linewidth=0.8)
    plt.plot(emulator_wavelengths, rdn[best_emu,:], c='green', linewidth=0.8)
    pointstr_modtran = ''
    pointstr_emu = ''
    for point_ind, pn in enumerate(point_names):
        pointstr_modtran += pn + ': {}\n'.format(points[best_modtran,point_ind])
        pointstr_emu += pn + ': {}\n'.format(points[best_emu,point_ind])

    plt.text(1000, 10 , pointstr_modtran, verticalalignment='top')
    plt.text(2000, 10 , pointstr_emu, verticalalignment='top')

    plt.savefig(os.path.join(args.fig_dir, 'rdn_plots_best_matches.png'),dpi=200,bbox_inches='tight')



    cmap = plt.get_cmap('coolwarm')
    for dim in range(points.shape[-1]):
        #slice = np.take(points,np.arange(0,points.shape[0]),axis=dim)
        slice = points[:,dim]
        un_vals = np.unique(slice)
        print(point_names[dim])

        for _val, val in enumerate(un_vals):
            loc_rdn = np.mean(rdn[slice == val, :], axis=0)
            loc_rdn_var = np.std(rdn[slice == val, :], axis=0)
            loc_rdn_atm = np.mean(rdn_atm[slice == val, :], axis=0)

            plt.plot(emulator_wavelengths, ref_rdn, c='black', linewidth=0.8)
            plt.plot(emulator_wavelengths, loc_rdn, c=cmap(float(_val)/len(un_vals)), linewidth=0.8)
            plt.fill_between(emulator_wavelengths, np.min(rdn[slice==val,:],axis=0), np.max(rdn[slice==val,:],axis=0), alpha=0.5, facecolor=cmap(float(_val)/len(un_vals)))
            #plt.plot(simulated_wavelengths, loc_rdn_atm, c=cmap(float(_val)/len(un_vals)), ls='--', linewidth=0.8)

        plt.ylim([0,11])

        pointstr = '{}: {} - {}'.format(point_names[dim], un_vals[0],un_vals[-1])
        plt.text(2000, 8 , pointstr, verticalalignment='top')

        plt.savefig('{}/dim_{}.png'.format(args.fig_dir, dim), dpi=200, bbox_inches='tight')
        plt.clf()


    for dim in range(points.shape[-1]):
        #slice = np.take(points,np.arange(0,points.shape[0]),axis=dim)
        slice = points[:,dim]
        un_vals = np.unique(slice)
        print(point_names[dim])

        for _val, val in enumerate(un_vals):
            loc_rdn = np.mean(rdn_modtran[slice == val, :], axis=0)
            loc_rdn_var = np.std(rdn_modtran[slice == val, :], axis=0)

            plt.plot(emulator_wavelengths, ref_rdn, c='black', linewidth=0.8)
            plt.plot(emulator_wavelengths, loc_rdn, c=cmap(float(_val)/len(un_vals)), linewidth=0.8)
            plt.fill_between(emulator_wavelengths, np.min(rdn_modtran[slice==val,:],axis=0), np.max(rdn_modtran[slice==val,:],axis=0), alpha=0.5, facecolor=cmap(float(_val)/len(un_vals)))
            #plt.plot(simulated_wavelengths, loc_rdn_atm, c=cmap(float(_val)/len(un_vals)), ls='--', linewidth=0.8)

        plt.ylim([0,11])

        pointstr = '{}: {} - {}'.format(point_names[dim], un_vals[0],un_vals[-1])
        plt.text(2000, 8 , pointstr, verticalalignment='top')

        plt.savefig('{}/modtran_dim_{}.png'.format(args.fig_dir, dim), dpi=200, bbox_inches='tight')
        plt.clf()



    #print_keys = ['Total\nTransmittance', 'Atmospheric Path\nReflectance', 'Spherical Albedo']
    print_keys=['rhoatm','sphalb','transm_down_dir','transm_down_dif', 'transm_up_dir','transm_up_dif'] 
    n_bands = int(modtran_results.shape[-1]/len(keys))

    fig = plt.figure(figsize=(20, 20 * 2 / (0.5 + len(print_keys))))
    gs = gridspec.GridSpec(ncols=len(keys), nrows=3, wspace=0.3, hspace=0.4)

    for key_ind, key in enumerate(keys):

        ax = fig.add_subplot(gs[0, key_ind])
        band_range = np.arange(n_bands * key_ind, n_bands * (key_ind + 1))
        indices = [test,band_range]

        plt.plot(emulator_wavelengths,np.median(d2_subset(modtran_results,indices),axis=0),c='red')
        #plt.plot(np.mean(d2_subset(sixs_results,indices),axis=0),c='green')

        plt.plot(emulator_wavelengths, np.median(pred[:,band_range], axis=0), c='blue', ls='--')

        if key_ind == 1:
            plt.legend(['Modtran','Emulator'])
        
        plt.title(print_keys[key_ind])
        point_names = npzf['point_names']
        #plt.xlabel('Wavelength [nm]')
        if key_ind == 0:
            plt.ylabel('Modeled Output')


    for key_ind, key in enumerate(keys):

        ax = fig.add_subplot(gs[1, key_ind])
        band_range = np.arange(n_bands * key_ind, n_bands * (key_ind + 1))
        indices = [test, band_range]

        mean_rel_error = np.median(np.abs(d2_subset(modtran_results,indices) - pred[:,band_range]) / d2_subset(modtran_results,indices), axis=0)
        #mean_rel_error = np.mean(np.abs(d2_subset(modtran_results,indices) - pred[:,band_range]), axis=0)
        #mean_rel_sixs_error = np.mean(np.abs(d2_subset(modtran_results,indices) - d2_subset(sixs_results,indices)), axis=0)
        mean_rel_sixs_error = np.median(np.abs(d2_subset(modtran_results,indices) - d2_subset(sixs_results,indices)) / d2_subset(modtran_results,indices), axis=0)

        #plt.title(print_keys[key_ind])
        plt.plot(emulator_wavelengths, mean_rel_error, c='blue')
        plt.plot(emulator_wavelengths, mean_rel_sixs_error, c='green', ls='--')

        if key_ind == 1:
            plt.legend(['Emulator\nResidual','6s Residual'])
        lims = list(ax.get_ylim())
        lims[1] = min(lims[1],1)
        plt.ylim([0,lims[1]])
        #plt.xlabel('Wavelength [nm]')
        if key_ind == 0:
            plt.ylabel('Median Relative Residual')

    for key_ind, key in enumerate(keys):

        ax = fig.add_subplot(gs[2, key_ind])
        band_range = np.arange(n_bands * key_ind, n_bands * (key_ind + 1))
        indices = [test, band_range]

        #mean_rel_error = np.median(np.abs(d2_subset(modtran_results,indices) - pred[:,band_range]) / d2_subset(modtran_results,indices), axis=0)
        mean_rel_error = np.median(np.abs(d2_subset(modtran_results,indices) - pred[:,band_range]), axis=0)
        mean_rel_sixs_error = np.median(np.abs(d2_subset(modtran_results,indices) - d2_subset(sixs_results,indices)), axis=0)
        #mean_rel_sixs_error = np.median(np.abs(d2_subset(modtran_results,indices) - d2_subset(sixs_results,indices)) / d2_subset(modtran_results,indices), axis=0)

        #plt.title(print_keys[key_ind])
        plt.plot(emulator_wavelengths, mean_rel_error, c='blue')
        plt.plot(emulator_wavelengths, mean_rel_sixs_error, c='green', ls='--')

        if key_ind == 1:
            plt.legend(['Emulator\nResidual','6s Residual'])
        lims = list(ax.get_ylim())
        lims[1] = min(lims[1],1)
        plt.ylim([0,lims[1]])
        plt.xlabel('Wavelength [nm]')
        if key_ind == 0:
            plt.ylabel('Median Absolute Residual')



    plt.savefig('{}/mean_test_set.png'.format(args.fig_dir), bbox_inches='tight')
    plt.clf()

    #fig = plt.figure(figsize=(10, 10 * 2 / (0.5 + 2)))
    fig = plt.figure(figsize=(20, 20 * 2 / (0.5 + len(print_keys))))
    gs = gridspec.GridSpec(ncols=len(keys), nrows=2, wspace=0.3, hspace=0.4)

    error_mean = np.zeros(len(test))
    for key_ind, key in enumerate(keys):
        band_range = np.arange(n_bands * key_ind, n_bands * (key_ind + 1))
        indices = [test, band_range]
        loc_error = np.nansum(np.abs(d2_subset(modtran_results,indices) - pred[:,band_range]),axis=1)
        error_mean += loc_error / np.nanmax(loc_error)
    order = np.argsort(error_mean)
    #order = np.argsort(points[test,0])

    lims_main = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    lims_diff = [[0, 0.25], [0, 0.25], [0, 0.25], [0, 0.25], [0, 0.25], [0, 0.25]]

    #for row in range(np.sum(test)):
    for row_ind, row in enumerate(order[np.linspace(0,len(order)-1,20,dtype=int)].tolist()):
        n_bands = int(modtran_results.shape[-1]/len(keys))

        for key_ind, key in enumerate(keys):

            ax = fig.add_subplot(gs[0, key_ind])
            band_range = np.arange(n_bands * key_ind, n_bands * (key_ind + 1))
            indices = [test,band_range]

            plt.plot(emulator_wavelengths, d2_subset(modtran_results,indices)[row,:],c='red')
            plt.plot(emulator_wavelengths, d2_subset(sixs_results,indices)[row,:],c='green')

            plt.plot(emulator_wavelengths, pred[row,band_range], c='blue', ls='--')
            plt.title(print_keys[key_ind])
            plt.ylim(lims_main[key_ind])

        for key_ind, key in enumerate(keys):

            ax = fig.add_subplot(gs[1, key_ind])
            band_range = np.arange(n_bands * key_ind, n_bands * (key_ind + 1))
            indices = [test,band_range]

            #mean_rel_error = np.abs(d2_subset(modtran_results,indices)[row,:] - pred[row,band_range]) / d2_subset(modtran_results,indices)[row,:]
            mean_rel_error = np.abs(d2_subset(modtran_results,indices)[row,:] - pred[row,band_range])

            mean_rel_sixs_error = np.abs(d2_subset(modtran_results,indices)[row,:] - d2_subset(sixs_results,indices)[row,:])


            plt.title(print_keys[key_ind])
            plt.plot(emulator_wavelengths, mean_rel_error, c='blue')
            plt.plot(emulator_wavelengths, mean_rel_sixs_error, c='green', ls='--')

            lims = lims_diff[key_ind]
            #lims = list(ax.get_ylim())
            #lims[1] = min(lims[1],1)
            if key_ind == 1:
                pointstr = ''
                for point_ind, pn in enumerate(point_names):
                    pointstr += pn + ': {}\n'.format(points[test,:][row, point_ind])

                plt.text(200, 0.9*(lims[1]-lims[0])+lims[0], pointstr, verticalalignment='top')

            plt.ylim([0,lims[1]])

        plt.savefig('{}/test_set_{}.png'.format(args.fig_dir, row_ind), bbox_inches='tight')
        plt.clf()

if __name__ == '__main__':
    main()
