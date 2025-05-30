#Import
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import xgcm
import veros.core.density.nonlinear_eq2 as neq2
import OCAPE_functions_xarray as Of_xr



def heigth_anomaly_CoM(salt, temp, mean_s, mean_t, zt, z_v):

    h_b0_d_zv = neq2.nonlin2_eq_of_state_dyn_enthalpy(salt, temp, -z_v)
    h_b0_d_z = neq2.nonlin2_eq_of_state_dyn_enthalpy(salt, temp, -zt)
    h_b0_d_zv_salt = neq2.nonlin2_eq_of_state_dyn_enthalpy(salt, 283.0 - 273.15, -z_v)
    h_b0_d_z_salt = neq2.nonlin2_eq_of_state_dyn_enthalpy(salt, 283.0 - 273.15, -zt)
    h_b0_d_zv_temp = neq2.nonlin2_eq_of_state_dyn_enthalpy(35.0, temp, -z_v)
    h_b0_d_z_temp = neq2.nonlin2_eq_of_state_dyn_enthalpy(35.0, temp, -zt)

    h_b0_d_z_avg = neq2.nonlin2_eq_of_state_dyn_enthalpy(mean_s, mean_t, -zt)
    h_b0_d_zv_avg = neq2.nonlin2_eq_of_state_dyn_enthalpy(mean_s, mean_t, -z_v)
    h_b0_d_z_avg_salt = neq2.nonlin2_eq_of_state_dyn_enthalpy(mean_s,
                                                              283.0 - 273.15,
                                                              -zt)
    h_b0_d_zv_avg_salt = neq2.nonlin2_eq_of_state_dyn_enthalpy(mean_s,
                                                               283.0 - 273.15,
                                                               -z_v)
    h_b0_d_z_avg_temp = neq2.nonlin2_eq_of_state_dyn_enthalpy(35.0, mean_t, -zt)
    h_b0_d_zv_avg_temp = neq2.nonlin2_eq_of_state_dyn_enthalpy(35.0, mean_t, -z_v)

    h_d_zv = h_b0_d_z - h_b0_d_zv
    h_d_zv_avg = h_b0_d_z_avg - h_b0_d_zv_avg
    ksi = (h_d_zv - h_d_zv_avg)/neq2.grav

    ksi_temp = (h_b0_d_z_temp - h_b0_d_zv_temp - (h_b0_d_z_avg_temp - h_b0_d_zv_avg_temp))/neq2.grav
    ksi_salt = (h_b0_d_z_salt - h_b0_d_zv_salt - (h_b0_d_z_avg_salt - h_b0_d_zv_avg_salt))/neq2.grav

    return ksi, ksi_temp, ksi_salt


def mean_mld(prho, z_ref=-10, sig_dev=0.01, time_slice=slice(0,None)):
    dsig = prho - prho.sel(zt=z_ref, method='nearest')
    dsig_ds = dsig.to_dataset(name='density')
    dsig_ds = dsig_ds.sel(Time=time_slice)
    coords = {'Z': {'center': 'zt'}}
    grid = xgcm.Grid(dsig_ds, coords=coords, periodic=False)
    dsig_ds_target = np.linspace(dsig_ds.density.min(dim=("Time","zt","yt","xt")).to_numpy(),
                                dsig_ds.density.max(dim=("Time","zt","yt","xt")).to_numpy() , 
                                2000)
    dsig_dens_br, dsig_zt_br = xr.broadcast(dsig_ds.density, dsig_ds.zt)
    mld = grid.transform(dsig_zt_br, 'Z', dsig_ds_target, target_data=dsig_dens_br)
    mld = mld.sel(density=sig_dev, method='nearest')
    return mld

def mld(prho, z_ref=-10, sig_dev=0.01):
    dsig = prho - prho.sel(zt=z_ref, method='nearest')
    dsig_ds = dsig.to_dataset(name='density')
    coords = {'Z': {'center': 'zt'}}
    grid = xgcm.Grid(dsig_ds, coords=coords, periodic=False)
    dsig_ds_target = np.linspace(dsig_ds.density.min(dim=("zt","yt","xt")).to_numpy(),
                                dsig_ds.density.max(dim=("zt","yt","xt")).to_numpy() , 
                                2000)
    dsig_dens_br, dsig_zt_br = xr.broadcast(dsig_ds.density, dsig_ds.zt)
    mld = grid.transform(dsig_zt_br, 'Z', dsig_ds_target, target_data=dsig_dens_br)
    mld = mld.sel(density=sig_dev, method='nearest')
    return mld

def Nsqr_decomposed_Time(temp, dzw, salt, zt):
    dct_dzt = temp.diff('zt').values / dzw.isel(zw=slice(0,-1)).values[np.newaxis,:,np.newaxis, np.newaxis]
    dsa_dzt = salt.diff('zt').values / dzw.isel(zw=slice(0,-1)).values[np.newaxis,:,np.newaxis, np.newaxis]
    dct_dzt = np.concatenate([dct_dzt, dct_dzt[:, -1:, :, :]], axis=1)
    dct_dzw = xr.DataArray(dct_dzt, dims=('Time', 'zw', 'yt', 'xt'), coords={'Time': temp.Time, 'zw': dzw.zw, 'yt': temp.yt, 'xt': temp.xt})
    dsa_dzt = np.concatenate([dsa_dzt, dsa_dzt[:, -1:, :, :]], axis=1)
    dsa_dzw = xr.DataArray(dsa_dzt, dims=('Time', 'zw', 'yt', 'xt'), coords={'Time': temp.Time, 'zw': dzw.zw, 'yt': temp.yt, 'xt': temp.xt})
    drho_dct = neq2.nonlin2_eq_of_state_drhodT(temp, -zt)
    drho_dsa = neq2.nonlin2_eq_of_state_drhodS()
    
    alpha = -(1 / neq2.rho0) * drho_dct
    beta = (1 / neq2.rho0) * drho_dsa
    Nsqr_T = neq2.grav * alpha.values * dct_dzw
    Nsqr_S = -neq2.grav * beta * dsa_dzw
    N_sqr = Nsqr_T + Nsqr_S
    return Nsqr_T, Nsqr_S, N_sqr

def Nsqr_decomposed_back(temp, dzw, salt, zt):
    dct_dzt = temp.diff('zt').values / dzw.isel(zw=slice(0,-1)).values[:,np.newaxis, np.newaxis]
    dsa_dzt = salt.diff('zt').values / dzw.isel(zw=slice(0,-1)).values[:,np.newaxis, np.newaxis]
    dct_dzt = np.concatenate([dct_dzt, dct_dzt[ -1:, :, :]], axis=0)
    dct_dzw = xr.DataArray(dct_dzt, dims=('zw', 'yt', 'xt'), coords={'zw': dzw.zw, 'yt': temp.yt, 'xt': temp.xt})
    dsa_dzt = np.concatenate([dsa_dzt, dsa_dzt[-1:, :, :]], axis=0)
    dsa_dzw = xr.DataArray(dsa_dzt, dims=('zw', 'yt', 'xt'), coords={'zw': dzw.zw, 'yt': temp.yt, 'xt': temp.xt})
    drho_dct = neq2.nonlin2_eq_of_state_drhodT(temp, -zt)
    drho_dsa = neq2.nonlin2_eq_of_state_drhodS()
    
    alpha = -(1 / neq2.rho0) * drho_dct
    beta = (1 / neq2.rho0) * drho_dsa
    Nsqr_T = neq2.grav * alpha.values * dct_dzw
    Nsqr_S = -neq2.grav * beta * dsa_dzw
    N_sqr = Nsqr_T + Nsqr_S
    return Nsqr_T, Nsqr_S, N_sqr

def Nsqr_decomposed(temp, dzt, salt, zt, zw):
    dct_dzt = temp.diff('zt') / dzt.isel(zt=slice(1,None))
    dsa_dzt = salt.diff('zt') / dzt.isel(zt=slice(1,None))
    first_value_dct_dzt = dct_dzt.isel(zt=0).expand_dims(dim='zt', axis=0)
    first_value_dsa_dzt = dsa_dzt.isel(zt=0).expand_dims(dim='zt', axis=0)
    dct_dzt = xr.concat([first_value_dct_dzt, dct_dzt], dim='zt')
    dsa_dzt = xr.concat([first_value_dsa_dzt, dsa_dzt], dim='zt')

    dct_dzw = xr.DataArray(dct_dzt.values, dims=('zw', 'yt', 'xt'), coords={'zw': zw, 'yt': temp.yt, 'xt': temp.xt})
    dsa_dzw = xr.DataArray(dsa_dzt.values, dims=('zw', 'yt', 'xt'), coords={'zw': zw, 'yt': temp.yt, 'xt': temp.xt})
    drho_dct = neq2.nonlin2_eq_of_state_drhodT(temp, -zt)
    drho_dsa = neq2.nonlin2_eq_of_state_drhodS()
    
    alpha = -(1 / neq2.rho0) * drho_dct
    beta = (1 / neq2.rho0) * drho_dsa
    Nsqr_T = neq2.grav * alpha.values * dct_dzw
    Nsqr_S = -neq2.grav * beta * dsa_dzw
    N_sqr = Nsqr_T + Nsqr_S
    return Nsqr_T, Nsqr_S, N_sqr

def alpha_beta(temp, zt):
    drho_dct = neq2.nonlin2_eq_of_state_drhodT(temp, -zt)
    drho_dsa = neq2.nonlin2_eq_of_state_drhodS()
    
    alpha = -(1 / neq2.rho0) * drho_dct
    beta = (1 / neq2.rho0) * drho_dsa
    return alpha, beta

def OCAPE(ds_sal, ds_temp, M=50):
    "ds_sal = absolute salinity, ds_temp = conservative temperature (works with potential but less accurate)"
    # Define the new press_bis axis (50 points)
    press = np.linspace(0, (-ds_temp.zt).max(), M)
    press_broadcasted = xr.DataArray(
        np.broadcast_to(press, (len(ds_sal['xt']), len(ds_sal['yt']), M)),
        dims=['xt', 'yt', 'press_bis'],
        coords={'xt': ds_sal['xt'], 'yt': ds_sal['yt'], 'press_bis': press}
    )
    last_non_nan_depth = (-ds_temp.zt).where(~np.isnan(ds_temp)).max(dim='zt')
    press_broadcasted = press_broadcasted.where(press_broadcasted <= last_non_nan_depth, np.nan)
    CTofP=Of_xr.interpolate_xarray(ds_temp, press_broadcasted)
    ASofP=Of_xr.interpolate_xarray(ds_sal, press_broadcasted)

    # Reshape using expand_dims to prepare for broadcasting
    ASofP_expanded = ASofP.expand_dims({'press_bis_2': M}, axis=1)  # Shape becomes (M, 1)
    CTofP_expanded = CTofP.expand_dims({'press_bis_2': M}, axis=1)  # Shape becomes (M, 1)
    press_expanded = press_broadcasted.expand_dims({'press_bis_2': M}).rename({'press_bis': 'press_bis_2', 'press_bis_2': 'press_bis'}) # Shape becomes (1, M)
    # Compute the enthalpy matrix using broadcasting
    # enthalpy_matrix = gsw.energy.enthalpy(ASofP_expanded,CTofP_expanded, press_expanded)
    # enthalpy_matrix =enthalpy_matrix.T
    # enthalpy_from_vec=gsw.energy.enthalpy(ASofP, CTofP, press_broadcasted)

    enthalpy_matrix = neq2.nonlin2_eq_of_state_dyn_enthalpy(ASofP_expanded,CTofP_expanded, press_expanded)
    enthalpy_from_vec= neq2.nonlin2_eq_of_state_dyn_enthalpy(ASofP, CTofP, press_broadcasted)
    # Apply the function across all (lat, lon) combinations
    result = xr.apply_ufunc(
        Of_xr.compute_rpe_single_ignore_nan,
        enthalpy_matrix,
        input_core_dims=[["press_bis", "press_bis_2"]],
        output_core_dims=[[],["rows_result"], ["cols_result"]],
        vectorize=True,
        dask="allowed",  # Enables compatibility with lazy computation
        output_dtypes=[float,int,int],
    )

    rpe_all=result[0]
    iPE=enthalpy_from_vec.mean(dim='press_bis')
    reduced_APE = iPE-rpe_all
    return reduced_APE

def pycnocline_fab(salinity, temperature, H, zt, weight_surf, weight_xyz_t, land_mask) :
    z_v = - H/2
    z_bottom = temperature.zt.where(temperature.zt > -H).min()
    mask_H = (zt >= z_bottom) & (((1-land_mask)*zt).min(dim='zt')<= z_bottom)
    salinity = salinity.where(mask_H)
    temperature = temperature.where(mask_H)

    sigma2 = neq2.nonlin2_eq_of_state_rho(salinity, temperature, -z_v) + neq2.rho0 -1000
    mean_sigma2 = sigma2.where(sigma2.zt > -H).weighted(weight_xyz_t).mean(dim=("zt"))
    sigma2_bot = sigma2.sel(zt =z_bottom, method = 'nearest')
    depth_anomaly = ((sigma2) * (temperature.zt+H/2)).where(sigma2.zt > -H).weighted(weight_xyz_t).mean(dim=("zt",))
    #depth_anomaly = ((sigma2-mean_sigma2) * (temperature.zt)).weighted(weight_xyz_t).mean(dim=("zt",))

    delta_sigma2 = (sigma2.sel(zt=z_bottom, method='nearest')-sigma2.isel(zt=-1))
    # delta_sigma2 = delta_sigma2.ffill(dim="yt")
    # delta_sigma2 = delta_sigma2.bfill(dim="yt")
    #delta_sigma2 = delta_sigma2.where((delta_sigma2.yt < 0) | (delta_sigma2 >= 0.05), 0.05)
    pic_fab = H/2 * (1- np.sqrt(1 + 8./H * (depth_anomaly/delta_sigma2)))
    meridional_pic_fab = pic_fab.weighted(weight_surf).mean(dim=("xt",))

    depth_anomaly_approx = -depth_anomaly/neq2.rho0
    mean_N2_approx = delta_sigma2 * neq2.grav / (H * neq2.rho0)
    zonavg_depth_anomaly_approx = depth_anomaly_approx.weighted(weight_surf).mean(dim=("xt",))
    zonavg_mean_N2_approx = mean_N2_approx.weighted(weight_surf).mean(dim=("xt",))

    H_plus = H * (1- np.sqrt(1 + 2./H * depth_anomaly/delta_sigma2 - (sigma2_bot-mean_sigma2)/delta_sigma2 ))
    H_plus = H_plus.weighted(weight_surf).mean(dim=("xt",))
    
    H_thin = (2 * depth_anomaly/delta_sigma2).weighted(weight_surf).mean(dim=("xt",))
    
    return meridional_pic_fab, zonavg_depth_anomaly_approx, zonavg_mean_N2_approx, H_plus, H_thin

def pycnocline_gnanadesikan(salinity, temperature, H, zt, weight_surf, weight_xyz_t, land_mask):
    z_v = - H/2
    z_bottom = temperature.zt.where(temperature.zt > -H).min()
    mask_H = (zt >= z_bottom) & (((1-land_mask)*zt).min(dim='zt')<= z_bottom)
    salinity = salinity.where(mask_H)
    temperature = temperature.where(mask_H)

    sigma2 = neq2.nonlin2_eq_of_state_rho(salinity, temperature, -z_v) + neq2.rho0 -1000
    delta_sigma2_gna = -sigma2 + sigma2.sel(zt =z_bottom, method = 'nearest')

    numer = (zt * delta_sigma2_gna).where(sigma2.zt > -H).weighted(weight_xyz_t).mean(dim=('zt'))
    denum = (delta_sigma2_gna).where(sigma2.zt > -H).weighted(weight_xyz_t).mean(dim=('zt'))
    pic_gna = -numer / denum
    meridional_pic_gna = pic_gna.weighted(weight_surf).mean(dim=("xt",))

    zonavg_numer = -numer.weighted(weight_surf).mean(dim=("xt",))
    zonavg_denum = denum.weighted(weight_surf).mean(dim=("xt",))
    return meridional_pic_gna, zonavg_numer, zonavg_denum
