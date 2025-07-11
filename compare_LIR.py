from My_Plugin.quantity import L_IR

sims = {
    'm12i_cd_reduced' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd/snap_600/run_SKIRT_i00_sed.dat',
    'm12i_cd_LTE' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_LTE/snap_600/run_SKIRT_i00_sed.dat',
    'm12i_cd_standard' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_standard/snap_600/run_SKIRT_i00_sed.dat',
    'm12i_cd_noiter' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_noiter/snap_600/run_SKIRT_i00_sed.dat',
}

for sim, path in sims.items():
    print(f"{sim}: ", end="")
    L_IR_value = L_IR(path)
    if L_IR_value is not None:
        print(f"{L_IR_value:.2e}")
    else:
        print("Data not available")

