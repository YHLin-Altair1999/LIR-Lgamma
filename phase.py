import yt
from My_Plugin.Add_Fields import add_fields
from My_Plugin.Cmaps import get_cmap
from My_Plugin.LoadData import unit_base

# Load the dataset.
path = '/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_600/'
ds = yt.load(path)
ds = add_fields(ds)

# Create a sphere of radius 100 kpc in the center of the domain.
c = [29345.283788, 30997.078816, 32484.066899]
my_sphere = ds.sphere(c, (100.0, "kpc"))

# Create a PhasePlot object.
# Setting weight to None will calculate a sum.
# Setting weight to a field will calculate an average
# weighted by that field.
plot = yt.PhasePlot(
    my_sphere,
    ("gas", "density"),
    ("gas", "temp"),
    ("gas", "mass"),
    weight_field=None,
)

# Set the units of mass to be in solar masses (not the default in cgs)
plot.set_unit(("gas", "mass"), "Msun")

# Save the image.
# Optionally, give a string as an argument
# to name files with a keyword.
plot.save()
