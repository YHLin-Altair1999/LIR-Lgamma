import yt

#ds = yt.load('/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_600')
ds = yt.load('/tscc/lustre/ddn/scratch/yel051/snapshots/FIRE3_Sam/m12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5/output/snapdir_060/snapshot_060.hdf5')
[print(field) for field in ds.field_list]
#dd = ds.all_data()
#print(dd['PartType0', 'CosmicRayEnergy'].units)
