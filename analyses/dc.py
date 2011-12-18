"""
:mod:`dc` -- DC sweep Analysis
------------------------------

.. module:: dc
.. moduleauthor:: Carlos Christoffersen

*****   Untested   *****

"""

from __future__ import print_function
import numpy as np

from paramset import ParamSet
from analysis import AnalysisError, ipython_drop
import nodal as nd
from nodalAD import DCNodalAD
from fsolve import solve, NoConvergenceError
import matplotlib.pyplot as plt

class Analysis(ParamSet):
    """
    DC Sweep Calculation

    Calculates a DC sweep of a circuit using the nodal approach. Nodal
    voltages are saved after the analysis is complete.

    Convergence parameters for the Newton method are controlled using
    the global variables in ``.options``.

    Output variables are stored in circuit and terminals with the
    'dC_' prefix.

    One plot window is generated for each ``.plot`` statement. Use
    'dc' type for this analysis.

    After completion the analysis drops to an interactive shell if the
    ``shell`` global variable is set to ``True``
    """

    # antype is the netlist name of the analysis: .analysis tran tstart=0 ...
    anType = "dc"

    # Define parameters as follows
    paramDict = dict(
        device = ('Instance name of device to sweep variable', '', str, ''),
        param = ('Device parameter to sweep', '', str, ''),
        start = ('Sweep start value', 'V', float, 0.),
        stop = ('Sweep stop value', 'V', float, 0.),
        sweep_num = ('Number of points in sweep', '', int, 50),
        verbose = ('Show iterations for each point', '', bool, False),
        fullAD = ('Use CPPAD for entire nonlinear part', '', bool, False)
        )


    def __init__(self):
        # Just init the base class
        ParamSet.__init__(self, self.paramDict)


    def run(self, circuit):
        """
        Calculates a DC sweep by solving nodal equations

        The parameter to be swept is specified in the analysis options
        """
        # for now just print some fixed stuff
        print('******************************************************')
        print('                 DC sweep analysis')
        print('******************************************************')
        if hasattr(circuit, 'title'):
            print('\n', circuit.title, '\n')

        # Only works with flattened circuits
        if not circuit._flattened:
            circuit.flatten()
            circuit.init()

        # get device 
        try:
            dev = circuit.elemDict[self.device]
        except KeyError: 
            raise AnalysisError('Could not find: {0}'.format(self.device))
            return

        paramunit = None
        if self.param:
            try:
                pinfo = dev.paramDict[self.param]
            except KeyError:
                raise AnalysisError('Unrecognized parameter: ' 
                                    + self.param)
            else:
                if not pinfo[2] == float:
                    raise AnalysisError('Parameter must be float: ' 
                                        + self.param)
        else:
            raise AnalysisError("Don't know what parameter to sweep!")

        # Create nodal object: for now assume devices do not change
        # topology during sweep
        nd.make_nodal_circuit(circuit)
        if self.fullAD:
            dc = DCNodalAD(circuit)
        else:
            dc = nd.DCNodal(circuit)
        x = dc.get_guess()

        pvalues = np.linspace(start = self.start, stop = self.stop, 
                              num = self.sweep_num)
        circuit.dC_sweep = pvalues
        circuit.dC_var = 'Device: ' + dev.nodeName \
            + '  Parameter: ' + self.param
        circuit.dC_unit = pinfo[1]
        
        xVec = np.zeros((circuit.nD_dimension, self.sweep_num))
        for i, value in enumerate(pvalues):
            setattr(dev, self.param, value)
            # re-process parameters (topology must not change, for now at least)
            dev.process_params()

            sV = dc.get_source()
            # solve equations
            try: 
                (x, res, iterations) = solve(x, sV, dc)
            except NoConvergenceError as ce:
                print(ce)
                return
            # Save result
            xVec[:,i] = x
            if self.verbose:
                print('{0} = {1}'.format(self.param , value))
                print('Number of iterations = ', iterations)
                print('Residual = ', res)

        # Save results in nodes
        circuit.nD_ref.dC_v = np.zeros(self.sweep_num)
        for i,term in enumerate(circuit.nD_termList):
            term.dC_v = xVec[i,:]

        # Process output requests
        for outreq in circuit.outReqList:
            if outreq.type == 'dc':
                plt.figure()
                plt.grid(True)
                plt.xlabel('{0} [{1}]'.format(circuit.dC_var, pinfo[1]))
                for termname in outreq.varlist:
                    term = circuit.termDict[termname]
                    plt.plot(pvalues, term.dC_v, 
                             label = 'Term: {0} [{1}]'.format(
                            term.nodeName, term.unit)) 
                if len(outreq.varlist) == 1:
                    plt.ylabel(
                        'Term: {0} [{1}]'.format(term.nodeName, term.unit))
                else:
                    plt.legend()
        plt.show()

        ipython_drop(globals(), locals())






