import numpy as np
import matplotlib.pyplot as plt
import My_Plugin.skirt.dust_scaling as d
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'font.family': 'serif'
    })

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
x = np.linspace(7.0, 9.5, 500)
nO_nH_ratio = 10**(x - 12)
Z = 2.04e-9*10**x * 0.0134
#print(2.04e-9*10**(8.69))
# Linear y-axis plot
ax1.plot(x, d.dust_to_gas_ratio_RemyRuyer(nO_nH_ratio), label='Rémy-Ruyer et al. (2014)', color='C0')
ax1.set_ylabel(r'$M_{\rm d}/M_{\rm g}$ (linear)')
ax1.set_yscale('log')
ax1.legend()

# Log y-axis plot
ax2.plot(x, d.dust_to_gas_ratio_RemyRuyer(nO_nH_ratio) / Z, color='C0')
#ax2.set_yscale('log')
ax2.set_xlabel(r'$12 + \log_{10}(n_{\rm O}/n_{\rm H})$')
ax2.set_ylabel(r'$M_{\rm d}/M_{\rm Z}$')

# Set common properties
for ax in [ax1, ax2]:
    ax.set_xlim(x[0], x[-1])

plt.tight_layout()
fig.savefig('Remy2014.png', dpi=300)

