#!/usr/bin/env python

#####################################################################################
#
# Copyright 2022 Quantinuum
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#####################################################################################

"""Functions for simulating QVT circuits."""

from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.random import random_unitary


def qv_circuits(nqubits: int,
                ntrials: int = 1,
                depth: Optional[int] = None,
                perm_seed: Optional[int] = None,
                gate_seed: Optional[int] = None) -> list:
    """
    Return a list of square quantum volume circuits 

    (Mostly from QV circuit generator from qiskit 0.20.0, newer versions
    only use fixed number of SU(4) gates)

    Args:
        nqubits:        number of qubits
        ntrials:        number of random iterations
        depth:          number of SU(4) rounds (defaults to QV depth=N)
        perm_seed:      random seed for permuting qubits
        gate_seed:      random seed for gates

    Returns:
        qv_circs: list of lists of circuits for the qv sequences
        (separate list for each trial)
        qv_circs_nomeas: same as above with no measurements for the ideal
        simulation
        
    """
    if depth is None:
        depth = nqubits
        
    qubit_list = list(range(nqubits))
    
    perm_rng = np.random.default_rng(perm_seed)
    gate_rng = np.random.default_rng(gate_seed)

    circuits = []
    circuits_nomeas = []

    # go through for each trial
    for trial in range(ntrials):

        qr = QuantumRegister(nqubits, 'qr')
        qr2 = QuantumRegister(nqubits, 'qr')
        cr = ClassicalRegister(nqubits, 'cr')

        qc = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr2, cr)

        qc.name = f'qv_depth_{nqubits}_trial_{trial}'
        qc2.name = qc.name

        # First gate has no permutations
        perm = list(range(nqubits))
        
        # build the circuit
        for _ in range(depth):
            # For each pair p in Pj, generate Haar random SU(4)
            for k in range(nqubits // 2):
                U = random_unitary(4, gate_rng).to_instruction()
                pair = int(perm[2*k]), int(perm[2*k+1])
                qc.compose(U, [qr[qubit_list[pair[0]]],
                              qr[qubit_list[pair[1]]]],
                              inplace=True)
                qc2.compose(U, [qr2[pair[0]],
                               qr2[pair[1]]],
                               inplace=True)
                               
            # Generate uniformly random permutation Pj of [0...n-1]
            perm = perm_rng.permutation(nqubits)

        # append an id to all the qubits in the ideal circuits
        # to prevent a truncation error in the statevector
        # simulators
        qc2.u1(0, qr2)

        circuits_nomeas.append(qc2)

        # add measurement
        for qind, qubit in enumerate(qubit_list):
            qc.measure(qr[qubit], cr[qind])

        circuits.append(qc)

    return circuits, circuits_nomeas


def convert(value:float,
            d: int,
            start: str,
            end: str) -> float:
    """
    Returns converted fidelity value.
    
    Args:
        value:   number to convert
        d:       Hilbert space dimension
        start:   quantity value measures
        end:     quantity to convert to
        
    Returns:
        converted value
        
    
    """
    d = float(d)
    
    if start == end:
        out = value
        
    elif start == 'avg' and end == 'dep':
        out = (d*value - 1)/(d - 1)
        
    elif start == 'avg' and end == 'proc':
        out = ((d + 1)*value - 1)/d
        
    elif start == 'dep' and end == 'avg':
        out = ((d - 1) *value + 1)/d
        
    elif start == 'dep' and end == 'proc':
        out = ((d**2 - 1) * value + 1)/d**2
        
    elif start == 'proc' and end == 'avg':
        out = (d * value + 1)/(d + 1)
        
    elif start == 'proc' and end == 'dep':
        out = (d**2 * value - 1)/(d**2 - 1)
        
    else:
        print(f'No equation for start = {start} and end = {end}')
        out = None
        
    return out

    
def binstr(n: int,
           nmax: int) -> str:
    """Returns binary string for integer n with nmax qubits."""

    return '0' * (nmax - len(bin(n)[2:])) + bin(n)[2:]
    

def gate_counts(qv_circs) -> dict:
    """Save all attributes of QVFitter object."""
    
    gcounts = {'u1': [], 'u2':[], 'u3':[], 'cx': [], 'cz':[]}
    for qc in qv_circs:
        try: 
            count_dict = qc.count_ops()
        except AttributeError:
            count_dict = QuantumCircuit.from_qasm_str(qc).count_ops()
        [
            gcounts[key].append(count_dict[key]) 
            for key in gcounts if key in count_dict
        ]
    return gcounts
    

def estimate_errors(error_dict: dict):
    """Estimate TQ errors based on error_dict."""

    tq_dep = 1
    sq_dep = 1
    if 'tq_dep' in error_dict:
        tq_dep *= convert(1 - error_dict['tq_dep'], 4, 'avg', 'dep')
        
    if 'tq_coh' in error_dict:
        tq_dep *= convert(1 - error_dict['tq_coh'], 4, 'avg', 'dep')
        
    if 'sq_dep' in error_dict:
        sq_dep *= convert(1 - error_dict['sq_dep'], 2, 'avg', 'dep')
        
    if 'sq_coh' in error_dict:
        sq_dep *= convert(1 - error_dict['sq_coh'], 2, 'avg', 'dep')

    if 'sq_dph' in error_dict:
        sq_dep *= convert(1 - error_dict['sq_dph'], 2, 'avg', 'dep')
    
    if 'tq_dph' in error_dict:
        tq_dep *= convert(1 - error_dict['tq_dph'], 2, 'avg', 'dep')
    
    sq_dep = convert(convert(sq_dep, 2, 'dep', 'proc') ** 2, 4, 'proc', 'dep')
    
    slice_fid = convert(sq_dep * tq_dep, 4, 'dep', 'avg')
    
    return slice_fid
    
    
def passing_error_estimate(qv_estimate,
                           nqubits: int,
                           threshold=2/3):
    """Returns estimated passing threshold of qv_estimate object for nqubits."""
    
    evals = list(qv_estimate.error_dict.keys())
    ndata = [np.mean(qv_estimate.act_success[nqubits, e]) for e in evals]
    cs = CubicSpline(evals, np.real(np.array(ndata) - threshold))
    
    return root_scalar(cs, x0=1e-4, x1=1e-3, method='secant').root


def convert_error_dict(error, etype):

    default_dict = {
        'sq_dep': 0, 
        'sq_coh':0, 
        'sq_dph': 0, 
        'tq_dep': 0, 
        'tq_coh': 0, 
        'tq_dph':0, 
        'tq_cross': 0, 
        'meas': 0,
        'prep': 0
    }
    
    edict = {
        'TQ Dep': {
            'tq_dep': 10*error/(12/5+10), 
            'sq_dep': error/(12/5+10), 
            'meas': error
        },
        'SQ Dep': {
            'sq_dep': 10*error/25, 
            'tq_dep': error/25, 
            'meas': error
        },
        'TQ Coh': {
            'tq_coh': 10*error/(12/5+10), 
            'sq_dep': error/(12/5+10), 
            'meas': error
        },
        'Measure': {
            'meas': 10*error, 
            'tq_dep': 10*error/(12/5+10), 
            'sq_dep': error/(12/5+10)
        },
        'Realistic cross': {
            'tq_cross': error,  
            'tq_dep': 10*error/(12/5+10), 
            'sq_dep': error/(12/5+10), 
            'meas': error
        },
        'Realistic dph': {
            'tq_dph': error, 
            'tq_dep': 10*error/(12/5+10), 
            'sq_dep': error/(12/5+10), 
            'meas': error
        },
        'Uncontrolled': {
            'tq_cross': 1e-3, 
            'tq_dph': error/2, 
            'tq_coh': error/(12/5+11), 
            'tq_dep': 10*error/(12/5+11), 
            'sq_dep': error/(12/5+11), 
            'meas': error
        },
        'Realistic coh': {
            'tq_coh': 5*error/(12/5+10), 
            'tq_dep': 5*error/(12/5+10), 
            'sq_dep': error/(12/5+10), 
            'meas': error
        },
        'Realistic all': {
            'tq_cross': error/2, 
            'tq_dph': error/2, 
            'tq_coh': error/(12/5+11), 
            'tq_dep': 10*error/(12/5+11), 
            'sq_dep': error/(12/5+11), 
            'meas': error
        }
    }
    out = {**default_dict, **edict[etype]}
   
    return out