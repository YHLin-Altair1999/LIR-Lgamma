import numpy as np
import pandas as pd
import yt
from glob import glob
import os
from My_Plugin.Add_Fields import add_fields

def load_snap(galaxy: str, snap_id: int) -> pd.DataFrame:
    base_name = f'/tscc/lustre/ddn/scratch/yel051/AHF_result/outputs/{galaxy}/snapshot_{snap_id:03d}.*.z*.*.AHF_halos'
    fnames = sorted(list(glob(base_name)))
    dfs = []
    for i, fname in enumerate(fnames):
        if i == 0:
            df = pd.read_csv(fname,
                             sep=r'\s+|\t+',    # split on one or more whitespace OR one or more tabs
                             engine='python',    # needed for regex separator
                             header=0,
                             skiprows=0)
            df.columns.str.replace('#', '').str.strip()
            columns = df.columns
        else:
            df = pd.read_csv(fname,
                             sep=r'\s+|\t+',    # split on one or more whitespace OR one or more tabs
                             engine='python',    # needed for regex separator
                             header=None,
                             names=columns,
                             skiprows=0)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    #df = df.sort_values(by='npart(5)', ascending=False)
    df = df.sort_values(by='M_star(65)', ascending=False)
    return df


def get_center(galaxy: str, snap_id: int) -> list:
    base_name = f'/tscc/lustre/ddn/scratch/yel051/AHF_result/outputs/{galaxy}/snapshot_{snap_id:03d}.*.z*.*.AHF_halos'
    fnames = sorted(list(glob(base_name)))
    #print(fnames)
    if len(fnames) != 0:
        df = load_snap(galaxy, snap_id)
        halo1 = df.iloc[0,:]
        out = list(halo1[['Xc(6)', 'Yc(7)', 'Zc(8)']])
        #print(out)
    else:
        print(f'AHF data not available, finding center by highest metallicity using yt.')
    '''
    path = get_snap_path(galaxy, snap_id)
    fnames = list(glob(os.path.join(path, '*.hdf5')))
    if len(fnames) == 1:
        path = fnames[0]
    ds = yt.load(path)
    ds = add_fields(ds)
    #v, c = ds.find_max(("PartType0", "Metallicity_00")) # use this as a proxy when there is no info from AHF
    v, c = ds.find_max(("PartType0", "Density")) # use this as a proxy when there is no info from AHF
    #v, c = ds.find_max(("gas", "CR_energy_density")) # use this as a proxy when there is no info from AHF
    print(f"Found center at {c}")
    out = [c[0].value, c[1].value, c[2].value]  # convert to list of floats
    '''
    return out

def get_radius(galaxy: str, snap_id: int) -> float:
    df = load_snap(galaxy, snap_id)
    halo1 = df.iloc[0,:]
    r = halo1['Rhalo(12)']
    return r

def get_angular_momentum(galaxy: str, snap_id: int) -> list:
    df = load_snap(galaxy, snap_id)
    halo1 = df.iloc[0,:]
    L_star = list(halo1[['Lx_star(68)', 'Ly_star(69)', 'Lz_star(70)']])
    #L_gas = list(halo1[['Lx_gas(48)', 'Ly_gas(49)', 'Lz_gas(50)']])
    return L_star

def get_snap_path(name: str, snap: int) -> str:
    paths = {
        'm12f_cd': f'/tscc/lustre/ddn/scratch/yul232/m12f_cr_700_mass_7000/output/snapdir_{snap:03d}/', # constant diffusion
        'm12i_hd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m12i_hd_fire3/output/snapdir_{snap:03d}/', # hydro
        'm12i_cd': f'/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_{snap:03d}/', # constant diffusion
        'm12i_et': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m12i_new_cr_runs/mode1_v1000_AlfC00/snapdir_{snap:03d}/', # extrinsic turbulence
        'm12i_sc_fx10': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m12i_new_cr_runs/mode6_v500_vAion_fx10/snapdir_{snap:03d}/', # self confinement
        'm12i_sc_fx100': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m12i_new_cr_runs/mode6_v500_vAion_fx100/snapdir_{snap:03d}/', # self confinement
        'm12r_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m12r_cr_700/output/snapdir_{snap:03d}/',
        'm12w_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m12w_cr_700/output/snapdir_{snap:03d}/',
        'm11f_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11f/m11f_cr_700_mass_7000/snapdir_{snap:03d}/',
        'm11f_sc_fcas50': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11f/m11f_sc_fcas50/snapdir_{snap:03d}/',
        'm11f_et_AlfvenMax': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11f/m11f_et_AlfvenMax/snapdir_{snap:03d}/',
        'm11f_et_FastMax': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11f/m11f_et_FastMax/snapdir_{snap:03d}/',
        'm11b_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11b_cr_700/output/snapdir_{snap:03d}/',
        'm11c_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11c_cr_700/output/snapdir_{snap:03d}/',
        'm11d_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11d_cr_700/output/snapdir_{snap:03d}/',
        'm11g_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11g_cr_700/output/snapdir_{snap:03d}/',
        'm11h_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11h_cr_700/output/snapdir_{snap:03d}/',
        'm11v_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m11v_cr_700/output/snapdir_{snap:03d}/',
        'm10v_cd': f'/tscc/lustre/ddn/scratch/yel051/snapshots/m10v_cr_700/output/snapdir_{snap:03d}/',
        'm09_cd' : f'/tscc/lustre/ddn/scratch/yel051/snapshots/m09_cr_700/output/snapdir_{snap:03d}/',
    }
    path = paths[name]
    #if len(list(glob(os.path.join(path, '*.hdf5')))) == 1:
    #    path += f'snapshot_{snap:03d}.hdf5'
    return path

if __name__ == '__main__':
    get_center('m12i_cd', 585)
