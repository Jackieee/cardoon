"""
:mod:`tlinpy4` -- Physical 4-terminal transmission line (Y-matrix based)
------------------------------------------------------------------------

.. module:: diode
.. moduleauthor:: Carlos Christoffersen

"""

import numpy as np
import circuit as cir
from globalVars import const, glVar
# For automatic differentiation:
#import cppaddev as ad

class Device(cir.Element):
    """
    4-terminal physical transmission line model using Y parameters::
    

             0 o===================================o 2
                               Z0
             1 o===================================o 3


    Code derived from fREEDA tlinp4 element. fREEDA implementation by
    Carlos E. Christoffersen, Mete Ozkar, Michael Steer

    Two models are supported dependent on the secting of nsect: When
    ``nsect = 0`` (not set) the frequency-domain model is enabled.
    When ``nsect > 0`` the transmission line is expanded in 
    ``nsect`` RLCG subsections.

    Netlist Examples::

      tlinpy4:tl1 in gnd out gnd z0mag=100. length=0.3m
      .model c_line tlinpy4 (z0mag=75.00 k=7 fscale=1.e10 alpha = 59.9)


    Internal Topology
    +++++++++++++++++

    The internal schematic is the following::
                 
          0 o----+------,               ,-----+-------o 2
       +         |      |               |     |              +
                ,-,     |               |    ,-, 
      v1        | |    /|\ y12 v2      /|\   | |             v2
            y11 | |   ( | )           ( | )  | | y22
       -        '-'    \V/      y21 v1 \V/   '-'             -
                 |      |               |     |  
          1 o----+------'               '-----+-------o 3

                       y11 = y22 , y12 = y21

    """

    # devtype is the 'model' name
    devType = "tlinpy4"
    numTerms = 4
    isFreqDefined = True
    fPortsDefinition = [(0, 1), (2, 3)]

    # Define parameters in a dictionary as follows: parameter name is
    # the key. Parameters are converted to class attributes after
    # circuit initialization.  If model is dependent on temperature,
    # include 'cir.Element.tempItem' ('temp' parameter description).
    paramDict = dict(
        k = ('Effective relative dielectric constant', '', float, 1.),
        alpha = ('Attenuation', 'dB/m', float, 0.1),
        z0mag = ('Magnitude of characteristic impedance', 'Ohms', float, 50.),
        fscale = ('Scaling frequency for attenuation', 'Hz', float, 0.),
        tand = ('Loss tangent', '', float, 0.),
        length = ('Line length', 'm', float, 0.1),
        nsect = ('Enable discrete approximation with n sections', '', int, 0),
        fopt = ('Optimum frequency for discrete approximation', 'Hz', float, 0)
        )

    def __init__(self, instanceName):
        # Here the Element constructor must be called. Do not connect
        # internal nodes here.
        cir.Element.__init__(self, instanceName)
        # Ambient temperature (temp) by default set to 27 C 
        # Add statements as needed


    def process_params(self, circuit):
        """
        Process parameters

        circuit: circuit instance that contains the element

        This should be called each time parameter values are changed.
        """
        # Called once the external terminals have been connected and the
        # non-default parameters have been set. Make sanity checks
        # here. Internal terminals/devices should also be defined here
        # (use circuit reference for this).  Raise cir.CircuitError if a fatal
        # error is found.

        # Use the following to make sure connections to internal
        # terminals are not repeated if this process_params is called
        # many times. 
        self.clean_internal_terms(circuit)
        # Calculate the capacitance per unit length
        # k = epsilon_eff
        self.c = np.sqrt(self.k) / (self.z0mag * const.c0)

        # Calculate the inductance per unit length
        self.l = (self.z0mag * self.z0mag) * self.c

        # convert alpha from db/m to nepers/m
        self.alpha_nepers = self.alpha / const.Np2dB

        # If nsect (number of sections) is set then use a
        # multi-section model for the transmisison line.  The strategy
        # is to expand the transmission line into a number of RLGC
        # sections. A circuit is built up with series L's and shunt
        # C's (which support series resistance and shunt capacitance)
        if self.nsect:
            # No longer a frequency-defined element
            self.isFreqDefined = False
            # Need to check that the local reference terminals are the
            # same as this is required by the sectional model.
            if self.neighbour[1] != self.neighbour[3]:
                raise cir.CircuitError('{1}: nsect > 0 but sectional model requires port references to be the same node'.format(self.nodeName)) 

            # Use discrete approximation.  Find the number of
            # subsections and the RLCG parameters of each.  First get
            # the length of each subsection.
            delta_x = self.length / self.nsect
            # Scale attenuation if fscale and fopt given
            if self.fscale and self.fopt:
                alphaf = self.alpha_nepers * np.sqrt(self.fopt / self.fscale)
            else:
                alphaf = self.alpha_nepers
            # Now get the R, L, G and C of each subsection.
            R = 2.0 * alphaf * self.z0mag * delta_x
            if self.fopt and self.tand:
                G = self.tand * 2. * np.pi * self.fopt * self.c * delta_x
            else:
                G = 0.

            L = self.l * delta_x
            C = self.c * delta_x

            # Create internal terminal list names
            gyrnode = self.nodeName + ':gyr'
            rnode = self.nodeName + ':r'
            cnode = self.nodeName + ':c'
            # Gyrated inductor cap
            indcap = L * glVar.gyr * glVar.gyr
            # Initialize to gnd node used for all gyrators (terminal 4)
            termlist = ['gnd']
            # nps: nodes added by one section
            if R:
                nps = 3
            else:
                nps = 2
            # Initialize empty lists
            self.linearVCCS = list()
            self.linearVCQS = list()
            # Expand transmission line sections
            for i in range(self.nsect):
                # input node number
                if i:
                    inn = nps*i + 4
                else:
                    inn = 0
                # add gyrator node
                termlist = [gyrnode + str(i)]
                # Gyrator node number
                gnn = nps*i + 5
                if R:
                    rnn = gnn + 1
                else:
                    rnn = gnn
                # The last section does not add cap node
                if i < self.nsect - 1:
                    # Add capacitor terminal
                    termlist.append(cnode + str(i))
                    cnn = rnn + 1
                else:
                    cnn = 2
                if R:
                    # add resistor node
                    termlist.append(rnode + str(i))
                    # Add gyrator
                    self.linearVCCS += [((inn, rnn), (4, gnn), glVar.gyr), 
                                        ((gnn, 4), (inn, rnn), glVar.gyr)]
                    # Add resistor
                    self.linearVCCS.append(((rnn, cnn), (rnn, cnn), 1./R))
                else:
                    # Add gyrator
                    self.linearVCCS += [((inn, cnn), (4, gnn), glVar.gyr), 
                                        ((gnn, 4), (inn, cnn), glVar.gyr)]
                # Add inductor
                self.linearVCQS.append(((gnn, 4), (gnn, 4), indcap))
                # Add capacitor
                self.linearVCQS.append(((cnn, 1), (cnn, 1), C))
                if G:
                    # Add conductor
                    self.linearVCLS.append(((cnn, 1), (cnn, 1), G))

        # Calculate temperature-dependent variables
        # self.set_temp_vars(self.temp)
        # if device is based on cppaddev, make sure tape is re-generated
        # ad.delete_tape(self)


    def get_OP(self, vPort):
        """
        Calculates operating point information
    
        Input:  vPort = [v1, v2]

        Output: dictionary with OP variables

        The frequency-domain model is always used for this calculation.
        """
        ydc = self.get_dc_ymatrix()
        iout = np.dot(ydc, vPort)
        self.OP = dict(
            V1 = vPort[0],
            V2 = vPort[1],
            I1 = iout[0],
            I2 = iout[1]
            )
        return self.OP

        

    #---------------------------------------------------------------------
    # Noise: in general requires a previous call to get_OP 
    #---------------------------------------------------------------------
    def get_noise(self, freq):
        """
        Return noise spectral density at frequency freq
        
        freq may be a scalar/vector
        Not implemented
        """
        pass

    #---------------------------------------------------------------
    # Linear frequency-defined device methods (isFreqDefined = True)
    #---------------------------------------------------------------
    def get_ymatrix(self, freq):
        """
        Returns a 3-D matrix with Y parameters

        Only used if nsect != 0

        freq: frequency vector/scalar. **Frequency can not be zero**
        """
        # The following calculation is vectorized
        omega = 2. * np.pi * freq

        if self.fscale:
            alphaf = self.alpha_nepers * np.sqrt(freq / self.fscale)
        else:
            alphaf = self.alpha_nepers
        r = 2.0 * alphaf * self.z0mag
        g = self.tand * omega * self.c
        # define imaginary unit
        j = complex(0., 1.)
        z1 = r + j * self.l * omega
        y1 = g + j * self.c * omega

        # characterestic impedance
        # Zo = sqrt( (R + jwL)/(G + jwC))
        Z0 = np.sqrt(z1 / y1)
        # attenuation
        # gamma = sqrt( (R + jwL)*(G + jwC))
        gamma = np.sqrt(z1 * y1)
         
        ctemp = gamma * self.length
        
        y11 = np.cosh(ctemp) / Z0 / np.sinh(ctemp)
        y12 = -1. / Z0 / np.sinh(ctemp)
        
        # return 3-D np.array
        return np.array([[y11, y12],
                         [y12, y11]])


    def get_dc_ymatrix(self):
        """
        Returns a matrix with the DC Y parameters

        Only used if nsect != 0

        The attenuation at DC is assumed to be alpha * length. This is
        probably too much but we do not want an infinity matrix
        """
        ctemp = self.alpha_nepers * self.length
        # Make sure attenuation is not too small
        ctempMin = 1e-5
        if ctemp < ctempMin:
            ctemp = ctempMin
        y11 = np.cosh(ctemp) / self.z0mag / np.sinh(ctemp)
        y12 = -1. / self.z0mag / np.sinh(ctemp)

        # Return DC Y matrix
        return np.array([[y11, y12],
                         [y12, y11]])



# Here you can add additional functions and classes that only are
# visible withing this module.
