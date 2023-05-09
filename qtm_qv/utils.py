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


"""Functions for simulating QVT circuits."""

import re


def gate_counts(qv_circs):
    """Save all attributes of QVFitter object."""
    
    gcounts = {'u1': [], 'u2':[], 'u3':[], 'cx': [], 'cz':[], 'rzz':[]}
    #for qc in qv_circs:
        # try: 
        #     count_dict = qc.count_ops()
        # except AttributeError:
        #     count_dict = QuantumCircuit.from_qasm_str(qc).count_ops()
        # [
        #     gcounts[key].append(count_dict[key]) 
        #     for key in gcounts if key in count_dict
        # ]
    str_id = re.compile(r'Rxxyyzz(.*) (.*);')


    for qc in qv_circs:
        # basic gates
        for g in gcounts:
            gcounts[g].append(qc.count(g))
        
        # for Rxxyyzz gates (H2-1 and H1-1 N > 16)
        ntq = 0
        qasm_lines = qc.split('\n')
        for line in qasm_lines:
            match = str_id.match(line)
            if match:
                vals = match.group(1).split(',')
                for val in vals:
                    if '0.0*pi' not in val:
                        ntq += 1

        gcounts['rzz'][-1] += ntq
            
    return gcounts
