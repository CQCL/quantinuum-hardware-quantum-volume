# Copyright 2022 Quantinuum (www.quantinuum.com)
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

from qiskit import QuantumCircuit


def gate_counts(qv_circs):
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
