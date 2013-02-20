"""
:mod:`expOP` -- Experimental Operating Point Analysis
-----------------------------------------------------

.. module:: expOP
.. moduleauthor:: Tapan Savalia, Carlos Christoffersen

"""


from __future__ import print_function
import numpy as np

from paramset import ParamSet
from analysis import AnalysisError, ipython_drop
from globalVars import glVar
#from fsolve import solve, NoConvergenceError
import nodalSP
import nodal

class DCOP(ParamSet):            
    """
    Experimental DC Operating Point
    -------------------------------

    Circuit equations are formulated for parallel computation, but for
    now parallel processing not implemented. Assumptions:

      * Circuit must be given already divided in subcircuit blocks:
        can not be already flattened.  This analysis must be performed
        before any other analysis (since all other analyses flatten
        the circuit hierarchy).

      * For now, the main circuit can reference a subcircuit
        definition only once.

    Calculates the DC operating point of all subcircuits using the
    nodal approach. After the analysis is complete, nodal voltages are
    saved in subcircuit block and terminals with the ``eND_``
    prefix. After this the analysis drops to an interactive shell if
    the ``shell`` global variable is set to ``True``.

    By default the voltage at all external voltages is printed after
    the analysis is complete. Optionally the operating points of
    nonlinear elements can be printed. 

    Example::

        .analysis eop intvars=1 shell=1

    """

    # antype is the netlist name of the analysis: .analysis tran tstart=0 ...
    anType = "eop"

    # Define parameters as follows
    paramDict = dict(
#        intvars = ('Print internal element nodal variables', '', bool, False),
#        elemop = ('Print element operating points', '', bool, False),
#        shell = ('Drop to ipython shell after calculation', '', bool, False)
        maxiter = ('Maximum number of Newton iterations', '', int, 100),
        maxdelta = ('Maximum change in deltax for Newton iterations', '', 
                    float, 50.),
        tol = ('Absolute tolerance for convergence', 'V', float, 1e-6)
        )

    def __init__(self):
        # Just init the base class
        ParamSet.__init__(self, self.paramDict)
        
    

    def run(self, circuit):
        """
        Calculates the operating point by solving nodal equations

        The state of all devices is determined by the values of the
        voltages at the controlling ports.
        """
        # for now just print some fixed stuff
        print('******************************************************')
        print('          Experimental Operating point analysis')
        print('******************************************************')
        if hasattr(circuit, 'title'):
            print('\n', circuit.title, '\n')

        # For now force dense matrices:
#        if glVar.sparse:
        if False:
            self.nd = nodalSP
        else:
            self.nd = nodal
            print('Using dense matrices\n')
        	
        # Initialize data structures for block calculations        
        self.init_blocks(circuit)
        # Set initial guess
        xnVec = np.empty(self.systemSize, dtype=float)
        xnVec[-circuit.nD_dimension:] = self.interconnectDCNodal.get_guess() 
        rowAcc = 0
        for dcno in self.dcList:
            dim = dcno.ckt.nD_dimension
            xnVec[rowAcc:rowAcc + dim] = dcno.get_guess()
            rowAcc += dim
        success = False
        # Begin Newton iterations
        for n in xrange(self.maxiter):
            deltax = self.get_deltax(xnVec)
            #print(deltax)
            # Do not allow updates greater than maxdelta
            delta = max(abs(deltax))
            print('delta = ', delta)
            if delta > self.maxdelta:
                deltax *= self.maxdelta/delta
            # Update x
            xnewVec = xnVec + deltax

            # Check if deltax is small enough
            n1 = np.all(abs(deltax) < self.tol)
            res = max(abs(deltax))
            if n1:
                success = True
                break
            # Save new result for next iteration
            xnVec[:] = xnewVec
            
        # Print results
        if success:
            print('Success!')
        else:
            print('No convergence')
        print('iterations = ', n+1)
        print('xnewVec = ', xnewVec)
    

    def init_blocks(self, circuit):
        """ 
        Initialize class attributes that are needed for subcircuit
        decomposition.

        List of attributes defined here:
        
        interconnectDCNodal: dc object for interconnect circuit
        systemSize: ....
        ...
        
        """
        # We want subcircuits arranged in alphabetical order
        nameList = [xsubckt.nodeName 
                    for xsubckt in circuit.subcktDict.itervalues()]
        nameList.sort()
        # To get subckt connections use circuit.subcktDict[name].neighbour
 
        # Before initializing the main circuit, flatten hierarchy of
        # subcircuits only. Also, create a list with all top-level
        # subcircuits (alhpabetical order).
        self.subcktList = []
        extInterconnectTerms = []
        for name in nameList:
            # get subcircuit definition
            subckt = circuit.cktDict[circuit.subcktDict[name].cktName] 
            self.subcktList.append(subckt)
            subckt.flatten()
            # Add external connections of each subcircuit to a list
            extInterconnectTerms += circuit.subcktDict[name].neighbour
        # Initialize main circuit including sub-circuits
        circuit.init() 
        
        # Prepare main circuit (interconnect) for nodal
        # analysis. Nodes connected to Sub1 go first, then nodes
        # connected to Sub2 and son on. The last nodes are internal
        # interconnect nodes not connected to any subcircuit.
        self.nd.make_nodal_circuit(circuit, extInterconnectTerms)
        self.interconnectDCNodal = self.nd.DCNodal(circuit) 
        sizeC = circuit.nD_dimension
        # systemSize: total dimension including subcircuits plus interconnect
        # (initially equal to sizeC, later incremented in loop)
        self.systemSize =  sizeC
        #import pdb; pdb.set_trace() 
        # Create list of nodal object for subcircuits
        self.dcList = []
        # shape interconnect blocks like M, N and C
        self.NList = []
        self.MList = []
        # Total number of interconnects in circuit
        self.nInterconnect = 0
        for subckt in self.subcktList:
            self.nd.make_nodal_circuit(subckt, subckt.get_connections())
            self.dcList.append(self.nd.DCNodal(subckt))
            nc = len(subckt.get_connections())
            Nj = np.zeros((subckt.nD_dimension, sizeC), dtype=float)
            np.fill_diagonal(Nj[0:nc, 
                                self.nInterconnect:self.nInterconnect+nc], -1.)
            self.NList.append(Nj)
            Mj = np.zeros((sizeC, subckt.nD_dimension), dtype=float)
            self.MList.append(Mj)
            self.nInterconnect += nc
            self.systemSize += subckt.nD_dimension
        # Create C
        self.C = np.zeros((sizeC, sizeC), dtype=float)
        np.fill_diagonal(self.C[0:self.nInterconnect, 0:self.nInterconnect], 1.)



    def get_deltax(self, xnVec):
        """
        Return deltax from:

        A deltax = sv - ivec

        """
        # ----------------------------------------------------------------
        # Process interconnect matrix block

        # xkVec is the 'inteconnect' part of the big vector. It includes
        # the currents that connect subcircuits
        xkVec = xnVec[-self.interconnectDCNodal.ckt.nD_dimension:]
        # Form interconnect nodal vector: different from xkVec because we
        # have to replace current source values with nodal voltages.
        xInt = xkVec.copy()
        counter = 0
        counter1 = 0
        for subckt in self.subcktList:
            nc = len(subckt.get_connections())
            xInt[counter1:counter1+nc] = xnVec[counter:counter+nc]
            counter += subckt.nD_dimension
            counter1 += nc
         # form a Jint Matrix : Jint[M1 M2 .....Mj C]   
        (iVec, Jint) = self.interconnectDCNodal.get_i_Jac(xInt)
        # svInt is the right-hand side vector for the interconnect
        svInt = self.interconnectDCNodal.get_source() - iVec
        # Add contributions of current sources (unknown for the nodal
        # object, so not included in iVec)
        svInt[0:self.nInterconnect] -= xkVec[0:self.nInterconnect]
        # Split Jacobian in blocks: J1, J2, ...
        counter = 0
        # Extract M1, M2 ...Mj from Jint
        for i,subckt in enumerate(self.subcktList):
            nc = len(subckt.get_connections())
            self.MList[i][:, 0:nc] = Jint[:, counter:counter+nc] 
            counter += nc
           # R = len(self.subcktList.get_internal_terminal())
        # Add elements to C
        self.C[:, counter: ] = Jint[:, counter: ]

     
        # solve this matrix:

        #         |A1          N1|   | x1  |   |  b1  |
        #         |    A2      N2|   | x2  |   |  b2  |
        #         |       .    : |   | :   | = |   :  |   (11)
        #	  |         .  : |   | xj  |   |   :  |
        #	  |M1  M2  ..  C |   |xkp1 |   | bkp1 |

        # Node numbering for circuit blocks: External(Interconnect)
        # nodes first then subcircuit internal nodes and so on...  

        # Get matrix blocks and source vectors from DCNodal objects
        AccC = self.C.copy()
        AList = []  # AList = [A1, A2,.....Aj] : Subcircuit List
        bList = []  # BList = [b1, b2......bj] : Source vector of subcircuit blocks
        rowAcc = 0                                              #------
        for dcno, Mj, Nj in zip(self.dcList, self.MList, self.NList):#|
            dim = dcno.ckt.nD_dimension                              #|
            xjVec = xnVec[rowAcc : rowAcc + dim]                     #|  xjVec : subcircuit source vectors (as shown in equation_11 : x1 to xj)
            rowAcc += dim                                            #|
            (iVec, Aj) = dcno.get_i_Jac(xjVec)                       #|
            AList.append(Aj)                                         #|    Equation__(14)
            bj = dcno.get_source() - iVec - np.dot(Nj, xkVec)        #|
            bList.append(bj)                                         #|
            Ainv = np.linalg.inv(Aj)                                 #|
            AccC -= np.dot(Mj, np.dot(Ainv, Nj))                     #|
            svInt -= np.dot(Mj, np.dot(Ainv, bj))                    #|
                                                                #------
       
        deltax = np.empty(self.systemSize, dtype=float)
        # Solve Eq. (14) to get nodal voltages at interconnect and
        # subcircuit external currents
        xkp1Vec = np.linalg.solve(AccC, svInt)
        deltax[-self.interconnectDCNodal.ckt.nD_dimension:] = xkp1Vec     # Interconnect source vectors
        
        rowAcc = 0                                  #----
        for Aj, bj, Nj in zip(AList, bList, self.NList):#|
            bj = bj - np.dot(Nj, xkp1Vec)                  #|
            xjVec = np.linalg.solve(Aj, bj) 		#|
            dim = bj.size 				#|  Eqaution_____(12)
            deltax[rowAcc:rowAcc + dim] = xjVec		#|
            rowAcc += dim				#|
            					     #----

        return deltax                                
        
aClass = DCOP

#         print('Number of iterations = ', iterations)
#         print('Residual = ', res)
# 
#         print('\n Node      |  Value               | Unit ')
#         print('----------------------------------------')
#         for key in sorted(circuit.termDict.iterkeys()):
#             term = circuit.termDict[key]
#             print('{0:10} | {1:20} | {2}'.format(key, term.nD_vOP, term.unit))
# 
#         if self.intvars or self.elemop:
#             for elem in circuit.nD_nlinElem:
#                 print('\nElement: ', elem.nodeName)
#                 if self.intvars:
#                     print('\n    Internal nodal variables:\n')
#                     for term in elem.get_internal_terms():
#                         print('    {0:10} : {1:20} {2}'.format(
#                                 term.nodeName, term.nD_vOP, term.unit))
#                 if self.elemop:
#                     print('\n    Operating point info:\n')
#                     for line in elem.format_OP().splitlines():
#                         print('    ' + line.replace('|',':'))
#         print('\n')
# 
#         def getvar(termname):
#             return circuit.termDict[termname].nD_vOP
# 
#         def getterm(termname):
#             return circuit.termDict[termname]
# 
#         def getdev(elemname):
#             return circuit.elemDict[elemname]
# 
#         if self.shell:
#             ipython_drop("""
# Available commands:
#     getvar(<terminal name>) returns variable at <terminal name>
#     getterm(<terminal name>) returns terminal reference
#     getdev(<device name>) returns device reference
# """, globals(), locals())
