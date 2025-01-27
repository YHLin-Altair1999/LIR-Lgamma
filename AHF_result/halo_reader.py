import pandas as pd
df = pd.read_csv('./outputs/snapshot_600.0000.z0.000.AHF_halos',
                 sep=r'\s+|\t+',    # split on one or more whitespace OR one or more tabs
                 engine='python',    # needed for regex separator
                 header=0,
                 skiprows=0)
df.columns = df.columns.str.replace('#', '').str.strip()
print(df.columns)
df.sort_values(by='Mhalo(4)')
halo1 = df.iloc[0,:]
print(halo1[['Xc(6)', 'Yc(7)', 'Zc(8)']])
print(halo1[['Lx(22)', 'Ly(23)', 'Lz(24)']])
