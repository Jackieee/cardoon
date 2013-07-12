"""
:mod:`diode` -- Diode model including charge and series resistance
------------------------------------------------------------------

.. module:: diode
.. moduleauthor:: Carlos Christoffersen

This model is based on PN Juction class, also defined here. The
junction class can be re-used for other models with PN junctions
"""

# use: import pdb; pdb.set_trace() for debuging

import numpy as np
from globalVars import const, glVar
import circuit as cir
import cppaddev as ad

#-----------------------------------------------------------------------
class Junction:
    """
    P-N Juction model.
    
    Based on carrot source, in turn based on spice/freeda diode
    model. This is intended to model any regular P-N junction such as
    Drain-Bulk, diodes, Collector-Bulk, etc.
    
    Transit time, breakdown and area effects not included
    """
    def process_params(self, isat, n, fc, cj0, vj, m, xti, eg0, Tnomabs):
        """
        Calculate variables dependent on parameter values only
        """
        # Saturation current
        self.isat = isat 
        # Emission coefficient
        self.n = n
        # Coefficient for forward-bias depletion capacitance
        self.fc = fc
        # Zero-bias depletion capacitance
        self.cj0 = cj0
        # Built-in junction potential
        self.vj = vj
        # PN junction grading coefficient
        self.m = m
        # Set some handy variables
        self._k1 = xti / self.n
        self._k2 = const.q * eg0 / self.n / const.k / Tnomabs
        self._k3 = const.q * eg0 / self.n / const.k
        self._k4 = 1. - self.m


    def set_temp_vars(self, Tabs, Tnomabs, vt, egapn, egap_t):
        """
        Calculate temperature-dependent variables
        """
        # Normalized temp
        tnratio = Tabs / Tnomabs
        self._t_is = self.isat * pow(tnratio, self._k1) \
            * np.exp(self._k2 - self._k3 / Tabs) 
        self._kexp = vt * self.n
        if self.cj0 != 0.:
            self._t_vj = self.vj * tnratio \
                - 3. * vt * np.log(tnratio) \
                - tnratio * egapn + egap_t
            self._t_cj0 = self.cj0 * (1. + self.m 
                                     * (.0004 * (Tabs - Tnomabs) 
                                         + 1. - self._t_vj / self.vj))
            self._k5 = self._t_vj * self._t_cj0 / self._k4
            self._k6 = self._t_cj0 * pow(1. - self.fc, - self.m - 1.)
            self._k7 = ((1. - self.fc * (1. + self.m)) * self._t_vj * self.fc 
                        + .5 * self.m * self._t_vj * self.fc * self.fc)

    def get_id(self, vd):
        """
        Returns junction current

        vd: diode voltage
        """
        # Regular junction current
        return self._t_is * (ad.safe_exp(vd / self._kexp) - 1.)

    def get_qd(self, vd):
        """
        Returns junction depletion charge

        vd: diode voltage
        """
        if self.cj0 != 0.:
            b = self.fc * self._t_vj - vd
            c = self._k5 * (1. - pow(1. - vd / self._t_vj, self._k4))
            d = self._k6 * ((1. - self.fc * (1. + self.m)) 
                            * vd + .5 * self.m * vd * vd / self._t_vj 
                            - self._k7) + self._k5 \
                            * (1. - pow(1. - self.fc, self._k4))
            return ad.condassign(b, c, d)
        else:
            return 0. * vd



#-----------------------------------------------------------------------
class Device(cir.Element):
    """
    Junction Diode
    --------------

    Based on the Spice model. Connection diagram::
    
               o  1                           
               |                            
             --+--
              / \     
             '-+-' 
               |                          
               o  0 

    Includes depletion and diffusion charges.

    Netlist examples::

        diode:d1 1 0 isat=10fA cj0=20fF

        # Electrothermal device
        diode_t:d2 2 3 1000 gnd cj0=10pF tt=1e-12 rs=100 bv = 4.

        # Model statement
        .model dmodel1 diode (cj0 = 10pF tt=1ps)

    Internal Topology
    +++++++++++++++++

    The internal representation is the following::

        0  o
           |
           \ 
           / Rs
           \ 
           / 
           |   Term : t2
           o---------,-------------,            
                     | i(vin)      |
          +         /|\          ----- q(vin)
        vin        | | |         -----
          -         \V/            |
                     |             |
        1  o---------'-------------'
                                  
    Terminal t2 not present if Rs = 0

    Important Note
    ++++++++++++++

    This implementation does not account for the power dissipation
    in Rs. Use an external thermal resistor if that is needed.
    """
    # Device category
    category = "Semiconductor devices"

    # devtype is the 'model' name
    devType = "diode"

    # Number of terminals. If numTerms is set here, the parser knows
    # in advance how many external terminals to expect. By default the
    # parser makes no assumptions and allows any number of connections
    #
    numTerms = 2
    
    # Create electrothermal device
    makeAutoThermal = True

    # Flags: at least one should be enabled for other than
    # linear R, L, C, M
    #
    isNonlinear = True
    vPortGuess = np.array([0.])

    # Define parameters (note most parameters defined in Junction)
    paramDict = dict(
        cir.Element.tempItem,
        isat = ('Saturation current', 'A', float, 1e-14),
        n = ('Emission coefficient', ' ', float, 1.),
        fc = ('Coefficient for forward-bias depletion capacitance', ' ', 
              float, .5),
        cj0 = ('Zero-bias depletion capacitance', 'F', float, 0.),
        vj = ('Built-in junction potential', 'V', float, 1.),
        m = ('PN junction grading coefficient', ' ', float, .5),
        tt = ('Transit time', 's', float, 0.),
        xti = ('Is temperature exponent', ' ', float, 3.),
        eg0 = ('Energy bandgap', 'eV', float, 1.11),
        tnom = ('Nominal temperature', 'C', float, 27.),
        ibv = ('Current at reverse breakdown voltage', 'A', float, 1e-10),
        bv = ('Breakdown voltage', 'V', float, np.inf),
        area = ('Area multiplier', ' ', float, 1.),
        rs = ('Series resistance', 'Ohms', float, 0.),
        kf = ('Flicker noise coefficient', '', float, 0.),
        af = ('Flicker noise exponent', '', float, 1.),
       )

    def __init__(self, instanceName):
        """
        Here the Element constructor must be called. Do not connect
        internal nodes here.
        """
        cir.Element.__init__(self, instanceName)
        self.jtn = Junction()

    def process_params(self, thermal = False):
        # Called once the external terminals have been connected and
        # the non-default parameters have been set. Make sanity checks
        # here. Internal terminals/devices should also be defined
        # here.  Raise cir.CircuitError if a fatal error is found.
        
        # Make sure tape is re-generated
        ad.delete_tape(self)
        # Remove internal terminals
        self.clean_internal_terms()
        # Set flag to add thermal ports if needed
        self.__addThermalPorts = True

        # Define topology first
        if self.rs != 0.:
            # Need 1 internal terminal
            t2 = self.add_internal_term('t2', 'V')
            g = self.area / self.rs
            self.linearVCCS = [((0, t2), (0, t2), g)]
            # Nonlinear device attributes
            self.csOutPorts = [(t2, 1)]
            self.noisePorts = [(t2, 1), (0, t2)]
            self.controlPorts = [(t2, 1)]
        else:
            # Nonlinear device attributes
            self.csOutPorts = [(0, 1)]
            self.noisePorts = [(0, 1)]
            self.controlPorts = [(0, 1)]

        self.qsOutPorts = [ ]
        self._qd = False
        if self.tt + self.cj0 != 0.:
            # Add charge source (otherwise the charge calculation is ignored)
            self.qsOutPorts = self.csOutPorts
            self._qd = True

        # Absolute nominal temperature
        self.Tnomabs = self.tnom + const.T0
        self.egapn = self.eg0 - .000702 * (self.Tnomabs**2) \
            / (self.Tnomabs + 1108.)

        # Calculate variables in junction
        self.jtn.process_params(self.isat, self.n, self.fc, self.cj0, self.vj, 
                                self.m, self.xti, self.eg0, self.Tnomabs)
        if not thermal:
            # Calculate temperature-dependent variables
            self.set_temp_vars(self.temp)


    def set_temp_vars(self, temp):
        """
        Calculate temperature-dependent variables for temp given in C
        """
        # Make sure tape is re-generated
        ad.delete_tape(self)
        # Absolute temperature
        Tabs = temp + const.T0
        # Thermal voltage
        self.vt = const.k * Tabs / const.q
        # Temperature-adjusted egap
        egap_t = self.eg0 - .000702 * (Tabs**2) / (Tabs + 1108.)
        self._Sthermal = 4. * const.k * Tabs * self.rs
        # Everything else is handled by the PN junction
        self.jtn.set_temp_vars(Tabs, self.Tnomabs, self.vt, self.egapn, egap_t)


    def eval_cqs(self, vPort):
        """
        Models intrinsic diode and charge.  

        vPort is a vector with 1 element (diode voltage)

        Returns one vector for current and another for charge. Charge
        vector is empty if both cj0 and tt are zero
        """
        # Calculate regular PN junction current and charge
        iD = self.jtn.get_id(vPort[0])

        if self.cj0 != 0.:
            qD = self.jtn.get_qd(vPort[0])
        else:
            qD = 0.

        if self.tt != 0.:
            qD += self.tt * iD

        # add breakdown current
        if (self.bv < np.inf):
            iD -= self.ibv * \
                ad.safe_exp(-(vPort[0] + self.bv) / self.n / self.vt)

        iD *= self.area
        idV = np.array([iD])
        if self._qd:
            qD *= self.area
            qdV = np.array([qD])
            return (idV, qdV)
        else:
            return (idV, np.array([]))


    def power(self, vPort, ioutV):
        """ 
        Calculate total instantaneous power 

        Input: control voltages and currents from eval_cqs()
        """
        return vPort[0] * ioutV[0]


    def get_OP(self, vPort):
        """
        Calculates operating point information

        Input:  vPort = [vd]
        Output: dictionary with OP variables
        """
        # First we need the Jacobian
        (outV, jac) = self.eval_and_deriv(vPort)

        # Save noise variables
        self._Sshot = 2. * const.q * outV[0]
        self._kFliker = self.kf * outV[0]

        # Use negative indexing for charge in case power is inserted
        # between current and charge.
        opDict = dict(
            VD = vPort[0],
            ID = outV[0],
            gd = jac[0,0],
            Sthermal = self._Sthermal,
            Sshot = self._Sshot,
            kFliker = self._kFliker
            )
        # Add capacitor
        if self._qd:
            opDict['Cd'] = jac[-1,0]

        return opDict
                   

    def get_noise(self, f):
        """
        Return noise spectral density at frequency f
        
        Requires a previous call to get_OP() 
        """
        sj = self._Sshot + self._kSflicker / pow(f, self.af)
        if self.rs != 0.:
            sV = np.array([sj, self._Sthermal])
        else:
            sV = np.array([sj])
        return sV


    # Use automatic differentiation for eval and deriv function
    eval_and_deriv = ad.eval_and_deriv
    eval = ad.eval



