from My_Plugin.quantity import L_IR

sims = {
    'm12i_cd' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd/snap_600/run_SKIRT_i00_sed.dat',
    'm12i_cd_LTE' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_LTE/snap_600/run_SKIRT_i00_sed.dat',
    #'m12i_cd_standard' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_standard/snap_600/run_SKIRT_i00_sed.dat',
    'm12i_cd_noiter' : '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_noiter/snap_600/run_SKIRT_i00_sed.dat',
    'm12i_cd_crc-convert_crcski': '/tscc/lustre/ddn/scratch/yel051/CRC_test/m12i_test_crcski/basic_MW_i00_sed.dat',
    'm12i_cd_crc-convert_myski': '/tscc/lustre/ddn/scratch/yel051/CRC_test/m12i_test/run_SKIRT_i00_sed.dat',
    'm12i_cd_OnlyYoung': '/tscc/lustre/ddn/scratch/yel051/SKIRT/output/m12i_cd_OnlyYoung/snap_600/run_SKIRT_i00_sed.dat',
}

for sim, path in sims.items():
    print(f"{sim}: ", end="")
    L_IR_value = L_IR(path).to('L_sun')
    if L_IR_value is not None:
        print(f"{L_IR_value:.2e}")
    else:
        print("Data not available")

