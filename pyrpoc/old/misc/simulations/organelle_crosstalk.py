import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

er_fluence = np.linspace(0, 100, 100)  
mito_fluence = np.linspace(0, 100, 100)  
ER, MITO = np.meshgrid(er_fluence, mito_fluence)

# Ca2+ release from ER
# sigmoidal function because of limited calcium concentration/release
def ca_release(er_fluence):
    return 1 / (1 + np.exp(-0.1 * (er_fluence - 50)))

# CHOP promotes apoptosis, it is an unfolfed protein response TF
# also hill-like
def chop_response(er_fluence):
    return 0.5 / (1 + np.exp(-0.08 * (er_fluence - 60)))

# caspase activation, non-hill model
def er_apoptosis(er_fluence):
    return 0.2 * (1 - np.exp(-0.03 * er_fluence))

# Ca²⁺ and CHOP amplify effect on mitochondira
# ca2+ induced permeability
# CHOP inhibits Bcl-2
def mito_damage(m_fluence, er_fluence):
    ca = ca_release(er_fluence)
    chop = chop_response(er_fluence)
    effective_fluence = m_fluence * (1 + 0.5 * ca + 0.5 * chop) # scaling factors to inhibitors, this is the cross talk term
    return 0.7 / (1 + np.exp(-0.1 * (effective_fluence - 40))) 

def total_apoptosis(m_fluence, er_fluence):
    mito_effect = mito_damage(m_fluence, er_fluence)
    er_effect = er_apoptosis(er_fluence)
    return np.clip(mito_effect + er_effect, 0, 1)


Z = np.zeros_like(ER)
for i in range(len(mito_fluence)):
    for j in range(len(er_fluence)):
        m = mito_fluence[i]
        e = er_fluence[j]
        Z[i, j] = total_apoptosis(m, e)
    

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(ER, MITO, Z, cmap='magma')
ax.set_xlabel('ER fluence')
ax.set_ylabel('mitochondria fluence')
ax.set_zlabel('apoptosis probability')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for m_level in [10, 30, 50, 70, 90]:
    death = total_apoptosis(m_level, er_fluence)
    ax.plot(er_fluence, death, label=f'mitochondria fluence = {m_level:.0f}')
ax.set_xlabel('ER fluence')
ax.set_ylabel('apoptosis robability')
ax.legend()
plt.tight_layout()
plt.show()
