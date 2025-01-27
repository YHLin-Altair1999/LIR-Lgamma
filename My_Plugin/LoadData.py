import numpy as np
import pandas as pd
import yt
from glob import glob

unit_base = {
    "UnitLength_in_cm": 3.09e21,
    "UnitVelocity_in_cm_per_s": 1e5,
    "UnitMass_in_g": 1.989e43,
}

def load_snap(snap_id: int) -> pd.DataFrame:
    base_name = f'/tscc/lustre/ddn/scratch/yel051/AHF_result/outputs/snapshot_{snap_id:03d}.*.z*.*.AHF_halos'
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


def get_center(snap_id: int) -> list:
    df = load_snap(snap_id)
    halo1 = df.iloc[0,:]
    out = list(halo1[['Xc(6)', 'Yc(7)', 'Zc(8)']])
    return out

def get_angular_momentum(snap_id: int) -> list:
    df = load_snap(snap_id)
    halo1 = df.iloc[0,:]
    out = list(halo1[['Lx_star(68)', 'Ly_star(69)', 'Lz_star(70)']])
    return out

if __name__ == '__main__':
    get_center(585)
