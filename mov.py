import numpy as np
import matplotlib.pyplot as plt
import yt
import os
import argparse
import My_Plugin.Fields as f
from My_Plugin.Add_Fields import add_fields
from My_Plugin.Cmaps import get_cmap
from My_Plugin.LoadData import unit_base, get_center, get_angular_momentum

parser = argparse.ArgumentParser(description="Run script with single or parallel processing.")
parser.add_argument('--np', type=int, help="Number of processes for parallel execution")
args = parser.parse_args()
if args.np:
    yt.enable_parallelism()
    num_procs = args.np
else:
    num_procs = 0

base_folder = '/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output'
folders = sorted([os.path.join(base_folder, name) for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))])
#folders = ['/tscc/lustre/ddn/scratch/yel051/snapshots/new_cr_runs/mode1_v1000_AlfC00/snap_060/cr_Apr2019_mode1_v1000_AlfC00_snapshot_060.hdf5']

plot_types = [
    #'FaceOn', 
    'EdgeOn', 
    #'projection', 
    #'slice'
             ]
normals = ['x', 'y', 'z']

fields = [
            #('gas', 'density'),
            #('gas', 'CR_energy_density'),
            #('gas', 'Internal_energy_density'),
            #('gas', 'epsilon_gamma'),
            #('gas', 'metal_density'),
            #('gas', 'Neutral_Hydrogen_Number_Density'),
            ('gas', 'Compton_y'),
            #('PartType0', 'Density'),
            #('PartType0', 'Pressure'),
            #('PartType0', 'Temperature'),
            #('PartType0', 'InternalEnergy'),
            #('PartType0', 'CosmicRayEnergy')
            ]

for folder in folders[-1:]:
#for folder in yt.parallel_objects(folders[0:], num_procs):
    ds = yt.load(folder)
    ds = add_fields(ds)
    #v, c = ds.find_max(("PartType0", "Metallicity_00")) # use this as a proxy when there is no info from AHF
    snap_id = int(folder.split('/')[-1].split('_')[-1])
    #snap_id = 60
    c = get_center(snap_id)
    L = np.array(get_angular_momentum(snap_id))
    L /= np.linalg.norm(L)
    new_z = L
    if abs(np.dot(new_z, [1, 0, 0])) < 0.9:
        new_x = np.cross([1, 0, 0], new_z)
    else:
        new_x = np.cross([0, 1, 0], new_z)
    new_x = new_x / np.linalg.norm(new_x)

    # Find new y-axis using cross product
    new_y = np.cross(new_z, new_x)
    for field in fields:
        for plot_type in plot_types:
            plots = []
            match plot_type:
                case 'slice':
                    for normal in normals:
                        p = yt.SlicePlot(ds, center=c, normal=normal, fields=field, width=(50, 'kpc'))
                        plots.append(p)
                case 'projection':
                    for normal in normals:
                        p = yt.ProjectionPlot(ds, center=c, normal=normal, fields=field, width=(50, 'kpc'))
                        plots.append(p)
                case 'FaceOn':
                    p = yt.OffAxisProjectionPlot(ds, center=c, normal=new_z, fields=field, north_vector=new_x, width=(150, 'kpc'))
                    plots.append(p)
                case 'EdgeOn':
                    p = yt.OffAxisProjectionPlot(ds, center=c, normal=new_x, fields=field, north_vector=new_z, width=(150, 'kpc'))
                    plots.append(p)
            for p in plots:
                p.set_cmap(field=field, cmap=get_cmap(field))
                p.annotate_timestamp(redshift=True)
                p.annotate_scale()
                #p.set_zlim(field, 1e-9, 5e-9)
                fname = str(ds) + '_' + plot_type
                p.save(name=f'./images/m12i/{field[1]}/{fname}')
