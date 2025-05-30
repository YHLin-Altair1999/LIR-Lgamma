from My_Plugin.quantity import L_IR

#m11d_full = L_IR('./output/m11d_cd/snap_600/run_SKIRT_i00_sed.dat')
m11d_MRN  = L_IR('./output/m11d_cd_noPAH/snap_600/run_SKIRT_i00_sed.dat')
m11d_LTE  = L_IR('./output/m11d_cd_LTE/snap_600/run_SKIRT_i00_sed.dat')
m11d_WD_MW  = L_IR('./output/m11d_cd_WD_MW/snap_600/run_SKIRT_i00_sed.dat')
m11d_WD_SMC  = L_IR('./output/m11d_cd_WD_SMC/snap_600/run_SKIRT_i00_sed.dat')

#print('m11d_full:', m11d_full)
print('m11d_LTE:', m11d_LTE)
print('m11d_WD_MW:', m11d_WD_MW)
print('m11d_WD_SMC:', m11d_WD_SMC)

