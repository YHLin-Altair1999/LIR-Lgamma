import numpy as np
import matplotlib.pyplot as plt
import yt
import os
import argparse
from glob import glob
import My_Plugin.Fields as f
from My_Plugin.Add_Fields import add_fields
from My_Plugin.Cmaps import get_cmap
from My_Plugin.LoadData import get_center, get_angular_momentum, get_snap_path, get_radius
from My_Plugin.zlims import get_zlim

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run script with single or parallel processing.")
    parser.add_argument('--np', type=int, help="Number of processes for parallel execution")
    args = parser.parse_args()
    if args.np:
        yt.enable_parallelism()
        return args.np
    return 0

def calculate_orientation_vectors(L):
    """Calculate the orientation vectors for the galaxy"""
    L = L / np.linalg.norm(L)
    new_z = L
    
    if abs(np.dot(new_z, [1, 0, 0])) < 0.9:
        new_x = np.cross([1, 0, 0], new_z)
    else:
        new_x = np.cross([0, 1, 0], new_z)
    new_x = new_x / np.linalg.norm(new_x)
    new_y = np.cross(new_z, new_x)
    
    return new_x, new_z

def generate_all_tasks():
    """Generate a list of all tasks to be processed"""
    tasks = []
    for galaxy in gal_snap_pairs.keys():
        snaps = gal_snap_pairs[galaxy]
        for snap in snaps:
            width = (0.5 * get_radius(galaxy, snap) / 2**0.5, 'kpc')
            for field in fields:
                for plot_type in plot_types:
                    if plot_type in ['slice', 'projection']:
                        for normal in normals:
                            tasks.append((galaxy, snap, field, plot_type, normal, width))
                    else:  # FaceOn or EdgeOn
                        tasks.append((galaxy, snap, field, plot_type, None, width))
    return tasks

def process_task(task):
    """Process a single visualization task"""
    try:
        galaxy, snap, field, plot_type, normal, width = task
        
        # Load dataset
        path = get_snap_path(galaxy, snap)
        fnames = list(glob(os.path.join(path, '*.hdf5')))
        if len(fnames) == 1:
            path = os.path.join(path, fnames[0])
        ds = yt.load(path)
        ds = add_fields(ds)
        
        # Get coordinates and orientation
        c = get_center(galaxy, snap)
        L = np.array(get_angular_momentum(galaxy, snap))
        new_x, new_z = calculate_orientation_vectors(L)
        
        # Create plot based on type
        if plot_type == 'slice':
            p = yt.SlicePlot(ds, center=c, normal=normal, fields=field, width=width)
        elif plot_type == 'projection':
            p = yt.ProjectionPlot(ds, center=c, normal=normal, fields=field, width=width)
        elif plot_type == 'FaceOn':
            p = yt.OffAxisProjectionPlot(
                ds, center=c, normal=new_z, fields=field, 
                north_vector=new_x, width=width, 
                #weight_field=('gas', 'density')
                )
        else:  # EdgeOn
            p = yt.OffAxisProjectionPlot(
                ds, center=c, normal=new_x, fields=field, 
                north_vector=new_z, width=width,
                #weight_field=('gas', 'density')
                )
        
        # Configure and save plot
        p.set_cmap(field=field, cmap=get_cmap(field))
        p.annotate_timestamp(redshift=True)
        p.annotate_scale()
        p.set_zlim(field, get_zlim(field)[0], get_zlim(field)[1])
        #p.set_zlim(field, 1e-9, 5e-9)
        
        fname = str(ds) + '_' + plot_type
        save_dir = f'./images/{galaxy}/{field[1]}'
        os.makedirs(save_dir, exist_ok=True)
        p.save(name=f'{save_dir}/{fname}')
        
    except Exception as e:
        print(f"Error processing task {task}: {str(e)}")

def main():
    # Configuration
    global plot_types, normals, fields, gal_snap_pairs, width
    gal_snap_pairs = {
        #'m12f_cd': [600],
        #'m12r_cd': [600],
        #'m12w_cd': [600],
        #'m12i_hd': [20] 
        'm12i_cd': [600], 
        #'m12i_et': [60], 
        #'m12i_sc_fx10': [60], 
        #'m12i_sc_fx100': [60],
        #'m11b_cd': [600],
        #'m11c_cd': [600],
        #'m11d_cd': [600],
        #'m11f_cd': [600],
        #'m11g_cd': [600],
        #'m11h_cd': [600],
        #'m11v_cd': [600],
        #'m10v_cd': [600],
        #'m09_cd': [600],
        #'m11f_et_AlfvenMax': [600],
        #'m11f_et_FastMax': [600],
        #'m11f_sc_fcas50': [600]
        }
    width = (40, 'kpc')
    plot_types = [
        'FaceOn', 
        'EdgeOn', 
        #'projection',
        #'slice'
    ]
    normals = ['x', 'y', 'z']
    fields = [
        #('gas', 'density'),
        ('gas', 'CR_energy_density'),
        #('gas', 'CRp_number_density'),
        ('gas', 'Pion_decay_gamma_ray_source_function'),
        ('gas', 'inverse_Compton_gamma_ray_source_function'),
        #('gas', 'u_B'),
        #('gas', 'Internal_energy_density'),
        #('gas', 'epsilon_gamma'),
        #('gas', 'metal_density'),
        #('gas', 'Neutral_Hydrogen_Number_Density'),
        ('gas', 'Compton_y'),
        #('gas', 'total_metallicity'),
        ('gas', 'gas_number_density')

    ]
    
    # Get number of processes
    num_procs = parse_arguments()
    
    # Generate all tasks
    all_tasks = generate_all_tasks()
    
    # Process all tasks in parallel
    for task in yt.parallel_objects(all_tasks, num_procs):
        print(task)
        process_task(task)

if __name__ == '__main__':
    main()
