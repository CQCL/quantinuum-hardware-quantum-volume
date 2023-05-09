# Copyright 2023 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' Functions for plotting quantum volume data from Quantinuum. '''

from typing import Optional

import numpy as np
from scipy.special import erf


def original_bounds(success: float,
                    trials: int):
    ''' Returns bounds from original CI method. '''

    sigma = np.sqrt(success*(1 - success)/trials)
    lower_ci = success - 2*sigma
    upper_ci = success + 2*sigma

    return lower_ci, upper_ci


def bootstrap_bounds(qv_fitter,
                     reps: int = 1000,
                     ntrials: Optional[int] = None):
    ''' Returns bounds from bootstrap CI method. '''

    nqubits = len(qv_fitter.qubit_lists[0])

    success = bootstrap(
        qv_fitter,
        reps,
        ntrials
    )
    qv_mean = np.mean([
        qv_fitter.heavy_output_counts[f'qv_depth_{nqubits}_trial_{i}']/qv_fitter._circ_shots[f'qv_depth_{nqubits}_trial_{i}']
        for i in range(ntrials)
    ])
    lower_ci = 2*qv_mean - np.quantile(success, 1/2 + erf(np.sqrt(2))/2)
    upper_ci = 2*qv_mean - np.quantile(success, 1/2 - erf(np.sqrt(2))/2)

    return lower_ci, upper_ci
    

def bootstrap(qv_fitter,
              reps: int = 1000,
              ntrials: Optional[int] = None):
    ''' 
    Semi-parameteric bootstrap QV data. 

    Notes:
        -updated to take arb number of shots per circuit
    
    '''
    nqubits = len(qv_fitter.qubit_lists[0])
        
    if not ntrials:
        ntrials = len(qv_fitter.heavy_output_counts)
        
    shot_list = np.array([
        qv_fitter._circ_shots[f'qv_depth_{nqubits}_trial_{i}']
        for i in range(ntrials)
    ])
    success_list = np.array([
        qv_fitter.heavy_output_counts[f'qv_depth_{nqubits}_trial_{i}']
        for i in range(ntrials)
    ])
    resampled_ind = np.random.randint(0, ntrials, size=(reps, ntrials))
    resampled_shots = shot_list[resampled_ind]
    resampled_probs = success_list[resampled_ind]/resampled_shots

    resampled_success = (
        np.sum(np.random.binomial(resampled_shots, resampled_probs), axis=1)
        / np.sum(resampled_shots, axis=1)
    )    
    return resampled_success