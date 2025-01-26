import yt

ds = yt.load('/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_600')
[print(field) for field in ds.field_list]
