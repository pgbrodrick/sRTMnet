



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
from matplotlib.lines import Line2D
from scipy import interpolate
import xarray as xr

def d2_subset(data,ranges):
    a = data.copy()
    a = a[ranges[0],:]
    a = a[:,ranges[1]]
    return a

def load(
    file: str,
    **kwargs,
) -> xr.Dataset:

    ds = xr.open_dataset(file, mode="r", lock=False, **kwargs)
    dims = ds.drop_dims("wl").dims

    # Create the point dimension
    ds = ds.stack(point=dims).transpose("point", "wl")
    ds.load()

    return ds

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('input_netcdf', type=str)
    parser.add_argument('--comparison_netcdf', type=str, default=None)
    parser.add_argument('--input_netcdf_name', type=str, default=None)
    parser.add_argument('--comparison_netcdf_name', type=str, default=None)
    parser.add_argument('--variables', type=str, nargs='+', default=['rhoatm','sphalb','transm_down_dir','transm_down_dif', 'transm_up_dir','transm_up_dif'])
    parser.add_argument('-fig_dir', type=str, default='figs')

    args = parser.parse_args()

    np.random.seed(13)

    rtm = load(args.input_netcdf)
    rtm_points = np.vstack([x.data.tolist() for x in rtm.point])
    if args.comparison_netcdf is not None:
        rtm_comp = load(args.comparison_netcdf)
        rtm_comp_points = np.vstack([x.data.tolist() for x in rtm.point])
        if np.all(rtm_points == rtm_comp_points) is False:
            print('Points do not match between input and comparison netcdf, terminating')
            exit()

    for name in list(args.variables):
        if name not in list(rtm.variables):
            print(f'Could not find {name} in primary .nc, terminating')
            exit()
        if args.comparison_netcdf and name not in list(rtm_comp.variables):
            print(f'Could not find {name} in secondary .nc, terminating')
            exit()
    


    point_names = list(rtm.point.to_index().names)

    bad_points = np.zeros(rtm_points.shape[0],dtype=bool)
    if 'surface_elevation_km' in point_names and 'observer_altitude_km' in point_names:
        bad_points = rtm_points[:, point_names.index('surface_elevation_km')]  >= rtm_points[:, point_names.index('observer_altitude_km')] -3
        bad_points[np.any(rtm_comp['transm_down_dif'].data[:,:] > 10,axis=1)] = True
        bad_points[np.any(rtm['transm_down_dif'].data[:,:] > 10,axis=1)] = True
    good_points = np.logical_not(bad_points)
    bad_ind = np.any(rtm_comp['transm_down_dif'].data[:,:] > 10,axis=1)

    rtm['transm_up_dif'].data[rtm['transm_up_dif'].data[:,:] > 1] = 0
    rtm['transm_up_dif'].data[rtm['transm_up_dif'].data[:,:] < 0] = 0
    rtm_comp['transm_up_dif'].data[rtm_comp['transm_up_dif'].data[:,:] > 1] = 0
    rtm_comp['transm_up_dif'].data[rtm_comp['transm_up_dif'].data[:,:] < 0] = 0

    rtm['sphalb'].data[:,rtm.wl > 1200][rtm['sphalb'].data[:,rtm.wl > 1200] > 0.1] = 0
    rtm['sphalb'].data[rtm['sphalb'].data[:,:] < 0] = 0

    print(np.sum(rtm_comp['sphalb'].data[:,rtm_comp.wl > 1200] > 0.1))
    subs = rtm_comp['sphalb'].data[:,rtm_comp.wl > 1200].copy()
    subs[subs > 0.1] = 0
    rtm_comp['sphalb'].data[:,rtm_comp.wl > 1200] = subs
    print(np.sum(rtm_comp['sphalb'].data[:,rtm_comp.wl > 1200] > 0.1))

    rtm_comp['sphalb'].data[rtm_comp['sphalb'].data[:,:] < 0] = 0

    lims_main = [[0, 1], [0, 0.25], [0, 0.35]]
    lims_diff = [[0, 0.25], [0, 0.1], [0, 0.1]]

    cmap = plt.get_cmap('coolwarm')
    for dim in range(len(point_names)):

        fig = plt.figure(figsize=(30, 10))
        gs = gridspec.GridSpec(ncols=len(args.variables), nrows=2,wspace=0.2, hspace=0.2)
        plt.suptitle(point_names[dim])

        #slice = np.take(points,np.arange(0,points.shape[0]),axis=dim)
        slice = rtm_points[:,dim]
        un_vals = np.unique(slice)

        for key_ind, key in enumerate(args.variables):
            ax = fig.add_subplot(gs[0,key_ind])
            plt.title(f'{args.input_netcdf_name}: {args.variables[key_ind]}')

            leg_lines = []
            leg_names = []
            for _val, val in enumerate(un_vals):
                rtm_slice = np.nanmean(rtm[key].data[np.logical_and(slice == val, good_points), :], axis=0)
                plt.plot(rtm.wl, rtm_slice, c=cmap(float(_val)/len(un_vals)), linewidth=1)

                leg_lines.append(Line2D([0], [0], color=cmap(float(_val)/len(un_vals)), lw=2))
                leg_names.append(str(round(val,2)))
            plt.grid()

            #plt.ylim(lims_main[key_ind])
            plt.xlabel('Wavelength [nm]')

            if key_ind == 1:

                if point_names[dim] == 'H2OSTR':
                    pn = 'Water Vapor\n[g cm$^{-1}$]'
                elif point_names[dim] == 'AOT550' or point_names[dim] == 'AERFRAC_2':
                    pn = 'Aerosol Optical\nDepth'
                elif point_names[dim] == 'observer_altitude_km':
                    pn = 'Observer\nAltitude [km]'
                elif point_names[dim] == 'surface_elevation_km':
                    pn = 'Surface\nElevation [km]'
                elif point_names[dim] == 'solar_zenith':
                    pn = 'Solar\nZenith Angle [deg]'
                else:
                    pn = point_names[dim]

                plt.legend(leg_lines, leg_names, title=pn)

                #pointstr = '{}: {} - {}'.format(point_names[dim], un_vals[0],un_vals[-1])
                #plt.text(50, 0.9 * (lims_main[key_ind][1] - lims_main[key_ind][0]) + lims_main[key_ind][0], pointstr,
                #         verticalalignment='top')
            #elif key_ind == 2 and args.comparison_netcdf:
            #     
            #    leg_lines = [Line2D([0], [0], color='black', lw=2),
            #                 Line2D([0], [0], color='black', lw=2, ls='--')
            #                ]
            #    plt.legend(leg_lines, [args.input_netcdf_name, args.comparison_netcdf_name], title='RTM')

        if args.comparison_netcdf:
            for key_ind, key in enumerate(args.variables):
                ax = fig.add_subplot(gs[1,key_ind])
                plt.title(f'{args.comparison_netcdf_name}: {key}')
                for _val, val in enumerate(un_vals):
                    rtm_comp_slice = np.nanmean(rtm_comp[key].data[np.logical_and(slice == val, good_points), :], axis=0)
                    plt.plot(rtm_comp.wl, rtm_comp_slice, c=cmap(float(_val)/len(un_vals)), linewidth=1)
                plt.grid()

        plt.savefig(f'{args.fig_dir}/dim_{point_names[dim]}.png', dpi=200, bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    main()
