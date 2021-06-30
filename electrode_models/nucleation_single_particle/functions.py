from scipy.integrate import solve_ivp

def residual(SV, SVdot, self, sep, counter, params):
    """
    Define the residual for the state of the single particle electrode.

    This is an array of differential and algebraic governing equations, one for each state variable in the anode (anode plus a thin layer of electrolyte + separator).

    1. The electric potential in the electrode phase is an algebraic variable.
        In the anode, phi = 0 is the reference potential for the system.
        In the cathode, the electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

        The residual corresponding to these variables (suppose an index 'j') are of the form:
            resid[j]  = (epression equaling zero)

    2. All other variables are governed by differential equations.
    
        We have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  
    
        The residuals corresponding to these variables will have the form:
            resid[j] = SVdot[j] - (expression equalling dSV/dt)
    """
    import numpy as np
    import cantera as ct
    
    # Initialize the residual:
    resid = np.zeros((self.nVars,))
    RbarT= cantera.gas_constant*self.elyte_obj.T
    # Save local copies of the solution vectors, pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']] # locates the SVptr in the entire solution vector.
    SVdot_loc = SVdot[SVptr['residual']] #is this the zero-ed matrix?
    Dr_dt = numpy.zeros_like(SV['Histogram'])
    # Read the electrode and electrolyte electric potential:
    phi_ed = SV_loc[SVptr['phi_ed']]
    phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]
    C_k_elyte = SV_loc[SVptr['C_k_elyte']]
    Histogram = SV_loc[SVptr['Histogram']]
    # Set electric potentials for Cantera objects:
    self.bulk_obj.electric_potential = phi_ed
    self.conductor_obj.electric_potential = phi_ed
    self.elyte_obj.electric_potential = phi_elyte
    
    # Faradaic current density is positive when electrons are consumed 
    # (Li transferred to the anode)
    sdot_electron_carbon = self.surf_obj.get_net_production_rates(self.bulk_obj)
    sdot_electron_cat = self.surf_obj.get_net_production_rates(self.cat_obj)
    sdot_electron = sdot_electron_carbon + sdot_electron_cat
    i_Far = -ct.faraday*sdot_electron
    
    a_d = (C_k_elyte['LiO2[elyt]']*N_a)**(-1./3.) # length scale of diffusion
    r_crit = 2.*self.gamma_surf*V/(RbarT*m.log(C_k_elyte['LiO2[elyt]']/self.c_liO2_sat*C_k_elyte['Li+[elyt]']/self.c_li_sat))  # m // critical radius
    #there's something about using cantera here for molar volume but I'm not sure
    N_crit = 4./3.*m.pi*r_crit**3.*ct.avogadro/V # number of molecules in the critical nucleus of size
    Del_G_Crit = self.phi*4./3.*m.pi*self.gamma_surf*r_crit**2. # J mol-1 // energy barrier of the nucleation
    if N_crit <0:
        Del_G_Crit =0
    Z = m.sqrt(Del_G_Crit/(phi*3*m.pi*k_B*T*N_crit)) # - // Zeldovich factor #forgot how to fix, DeCaluwe should commit his code
    V_crit = 4./3.*m.pi*r_crit**3. # m3 // Critical volume
    N_sites = SV[SVptr['Area']]/(m.pi*r_crit**2) # number of nucleation sites
    k_nuc= self.D_LiO2*(a_d**-2)
    DN_Dt = k_nuc*N_sites*Z*m.exp(-Del_G_Crit/(k_B*T)) # same as before
    for i, r in enumerate(self.radius):
        if r > r_crit:
            Dnp_dt[i] += DN_Dt
            break
    for i, N in enumerate(Histogram):
        Dr_dt[i] = self.D_LiO2*V*(C_k_elyte['Li+[elyt]']- self.c_li_sat)*(C_k_elyte['LiO2[elyt]']-self.c_liO2_sat)/(radii[i]+D_LiO2/k_surf)- m.pi*radii[i]**2*N*gamma_surf*k_surf_des
        dNdt_radii = Dr_dt[i]/self.bin_width*N
        if dNdt_radii <0:
            SVdot_loc[SVptr['Histogram']][i] += dNdt_radii
            if i > 0:
                SVdot_loc[SVptr['Histogram']][i-1] -= dNdt_radii
        elif dNdt_radii > 0 and radii[i] != radii[-1]:
            SVdot_loc[SVptr['Histogram']][i] -= dNdt_radii
            SVdot_loc[SVptr['Histogram']][i+1] += dNdt_radii
    # Double layer current has the same sign as i_Far:
    i_dl = self.i_ext_flag*params['i_ext']/self.A_surf_ratio - i_Far
    

    #Final 
    SVdot_loc[SVptr['C_k_elyte']['LiO2[elyt]]'] =  -SVdot_loc[SVptr['Histogram']]*V_crit/(V*Elyte_v_SI) - 2*np.sum(Dr_dt*radii*radii)*np.pi*V_crit/(V*Elyte_v_SI)
    SVdot_loc[SVptr['C_k_elyte']['Li+[elyt]]'] =  -SVdot_loc[SVptr['Histogram']]*V_crit/(V*Elyte_v_SI) - 2*np.sum(Dr_dt*radii*radii)*np.pi*V_crit/(V*Elyte_v_SI)
    SVdot_loc[SVptr['Area']] = - DN_Dt*m.pi*r_crit**2 - (4*np.pi*np.sum(radii*Dr_dt)*np.pi)

    if self.name=='anode':
        # The electric potential of the anode = 0 V.
        resid[SVptr['residual'][SVptr['phi_ed']]] = SV_loc[SVptr['phi_ed']]
    
    elif self.name=='cathode':
        # For the cathode, the potential of the cathode must be such that the 
        # electrolyte electric potential (calculated as phi_ca + dphi_dl) 
        # produces the correct ionic current between the separator and cathode:
        N_k_sep, i_io = sep.electrode_boundary_flux(SV, self, sep)               
        resid[SVptr['phi_ed']] = i_io - params['i_ext']

    # Differential equation for the double layer potential:
    resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv)

    # TEMPORARY: Set time derivatives for the species concentrations to zero:
    resid[SVptr['C_k_elyte']] = SVdot_loc[SVptr['C_k_elyte']]
    resid[SVptr['Area']] = SVdot_loc[SVptr['Area']]
    resid[SVptr['Histogram']] = SVdot_loc[SVptr['Histogram']]
    
    return resid

def voltage_lim(SV, self, val):
    """
    Check to see if the voltage limits have been exceeded.
    """
    # Save local copies of the solution vector and pointers for this electrode:
    SVptr = self.SVptr
    SV_loc = SV[SVptr['residual']]
    
    # Calculate the current voltage, relative to the limit.  The simulation 
    # looks for instances where this value changes sign (i.e. crosses zero)    
    voltage_eval = SV_loc[SVptr['phi_ed']] - val
    
    return voltage_eval

def adjust_separator(self, sep):
    """ 
    Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
    """
    # Return the separator class object, unaltered:
    return sep

"""Citations"""
#Official Soundtrack:
#   # Mamma Mia!/Mamma Mia! Here We Go Again: The Movie Soundtracks Featuring Songs of ABBA
#   # Dancing Queen - Cher