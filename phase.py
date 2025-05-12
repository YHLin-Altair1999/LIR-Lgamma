import yt
from My_Plugin.Add_Fields import add_fields
from My_Plugin.Cmaps import get_cmap
from My_Plugin.LoadData import unit_base, get_center, get_snap_path
import os
from glob import glob

#global plot_types, normals, fields, gal_snap_pairs, width
gal_snap_pairs = {
    'm12i_cd': [600], 
    'm12i_et': [60], 
    'm12i_sc_fx10': [60], 
    'm12i_sc_fx100': [60],
    'm11b_cd': [600],
    'm11c_cd': [600],
    'm11d_cd': [600],
    'm11f_cd': [600],
    'm11g_cd': [600],
    'm11h_cd': [600],
    'm11v_cd': [600],
    'm10v_cd': [600],
    'm09_cd': [600],
    'm11f_et_AlfvenMax': [600],
    'm11f_et_FastMax': [600],
    'm11f_sc_fcas50': [600]
    }
fields = [
        #('gas', 'density'),
        #('gas', 'CR_energy_density'),
        #('gas', 'CRp_number_density'),
        ('gas', 'Pion_decay_gamma_ray_source_function_in_cell'),
        ('gas', 'inverse_Compton_gamma_ray_source_function_in_cell'),
        #('gas', 'u_B'),
        #('gas', 'Internal_energy_density'),
        #('gas', 'epsilon_gamma'),
        #('gas', 'metal_density'),
        #('gas', 'Neutral_Hydrogen_Number_Density'),
        #('gas', 'Compton_y'),
    ]
width = (20, 'kpc')

def one_plot(galaxy, snap, 
    xfield=("gas", "gas_number_density"), 
    yfield=("gas", "temp"), 
    zfield=("gas", "Pion_decay_gamma_ray_source_function_in_cell")
    ):
    # Load the dataset.
    path = get_snap_path(galaxy, snap)
    fnames = list(glob(os.path.join(path, '*.hdf5')))
    if len(fnames) == 1:
        path = os.path.join(path, fnames[0])
    ds = yt.load(path)
    ds = add_fields(ds)
    c = get_center(galaxy, snap)

    my_sphere = ds.sphere(c, width)

    # Create a PhasePlot object.
    # Setting weight to None will calculate a sum.
    # Setting weight to a field will calculate an average
    # weighted by that field.
    plot = yt.PhasePlot(
        my_sphere, xfield, yfield, zfield, weight_field=None
    )
    plot.set_log(zfield, False)
    plot.set_cmap(zfield, 'inferno')
    #plot.set_ylim(1e-6, 2e4)
    #plot.set_xlim(1e-30, 1e-20)
    fname = str(ds)
    save_dir = f'./phase_diagrams/{galaxy}/{zfield[1]}'
    os.makedirs(save_dir, exist_ok=True)
    plot.save(name=f'{save_dir}/{fname}')

if __name__ == "__main__":
    for galaxy in gal_snap_pairs.keys():
        snaps = gal_snap_pairs[galaxy]
        for snap in snaps:
            for field in fields:
                one_plot(galaxy, snap, zfield=field)
