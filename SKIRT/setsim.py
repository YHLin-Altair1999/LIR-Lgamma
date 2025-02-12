import astropy.units as u
import astropy.constants as c
from My_Plugin.skirt.convert import convert_stellar, convert_gas
from My_Plugin.skirt.modify_ski import modify_skifile
import sys

template_ski = '/tscc/lustre/ddn/scratch/yel051/SKIRT/template_ski/template.ski'
target_ski = 'run_SKIRT.ski'
galaxy = str(sys.argv[2])#'m12i_et'
snap_id = int(sys.argv[1])
output_dir = f'output/{galaxy}/snap_{snap_id}'
box_size = 30*u.kpc

if __name__ == '__main__':
    modify_skifile(template_ski, target_ski, [box_size, box_size, box_size])
    convert_stellar(
        galaxy, snap_id=snap_id, r_max=3**0.5*box_size/2
        )
    convert_gas(
        galaxy, snap_id=snap_id, r_max=3**0.5*box_size/2
        )

