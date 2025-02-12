import numpy as np
import pandas as pd
import yt
from glob import glob
import os

unit_base = {
    "UnitLength_in_cm": 3.09e21,
    "UnitVelocity_in_cm_per_s": 1e5,
    "UnitMass_in_g": 1.989e43,
}

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
    df = df.sort_values(by='npart(5)', ascending=False)
    return df


def get_center(galaxy: str, snap_id: int) -> list:
    df = load_snap(galaxy, snap_id)
    halo1 = df.iloc[0,:]
    out = list(halo1[['Xc(6)', 'Yc(7)', 'Zc(8)']])
    return out

def get_angular_momentum(galaxy: str, snap_id: int) -> list:
    df = load_snap(galaxy, snap_id)
    halo1 = df.iloc[0,:]
    out = list(halo1[['Lx_star(68)', 'Ly_star(69)', 'Lz_star(70)']])
    return out

def get_snap_path(name: str, snap: int) -> str:
    paths = {
        'm12i_cd': f'/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_{snap:03d}/', # constant diffusion
        'm12i_et': f'/tscc/lustre/ddn/scratch/yel051/snapshots/new_cr_runs/mode1_v1000_AlfC00/snapdir_{snap:03d}/', # extrinsic turbulence
        'm12i_sc_fx10': f'/tscc/lustre/ddn/scratch/yel051/snapshots/new_cr_runs/mode6_v500_vAion_fx10/snapdir_{snap:03d}/', # self confinement
        'm12i_sc_fx100': f'/tscc/lustre/ddn/scratch/yel051/snapshots/new_cr_runs/mode6_v500_vAion_fx100/snapdir_{snap:03d}/' # self confinement
    }
    path = paths[name]
    #if len(list(glob(os.path.join(path, '*.hdf5')))) == 1:
    #    path += f'snapshot_{snap:03d}.hdf5'
    return path

if __name__ == '__main__':
    get_center('m12i_cd', 585)
