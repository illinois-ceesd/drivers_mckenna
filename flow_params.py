import numpy as np
import cantera as ct

fator = 1.0 #7.5 #1.0/0.117892786

print(fator)

r_int = 0.03
r_ext = 0.035


for rxnmech in ['uiuc_sharp']:

    gas = ct.Solution(rxnmech+".yaml")
    air = "O2:0.21,N2:0.79"
    gas.set_equivalence_ratio(phi=1.0, fuel="C2H4:1", oxidizer=air)

    gas.TP = 300, ct.one_atm

    rho_int = gas.density

    x = gas.X

print(x[0])
print(np.sum(x[1:]))

mass_unb = 25*fator/np.sum(x[1:])
mass_shr = 20*fator


A_int = np.pi*r_int**2
A_ext = np.pi*(r_ext**2 - r_int**2)

lmin_to_m3s = 1.66667e-5
u_int = mass_unb*lmin_to_m3s/A_int
u_ext = mass_shr*lmin_to_m3s/A_ext

rhoU_int = rho_int*u_int

print("V_dot=",mass_unb,"(L/min)")
print(f"{A_int= }","(m^2)")
print(f"{A_ext= }","(m^2)")
print(f"{u_int= }","(m/s)")
print(f"{u_ext= }","(m/s)")
print(f"{rhoU_int= }")
print("ratio=",u_ext/u_int)

mdot_int = rho_int*u_int*A_int

rho_ext = 101325/((8314/gas.molecular_weights[-1])*300.)
mdot_ext = rho_ext*u_ext*A_ext

print("mdot_air+fuel=",mdot_int*1000000,"(mg/s)")
print("mdot_shroud=",mdot_ext*1000000,"(mg/s)")

