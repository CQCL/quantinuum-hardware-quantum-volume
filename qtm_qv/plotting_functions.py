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

''' Functions for plotting quantum volume data from Quantinuum. '''

from typing import Optional


import numpy as np
import matplotlib.pyplot as plt

from qtm_qv.analysis_functions import bootstrap_bounds, original_bounds

ecolor = plt.get_cmap('tab10').colors


def success_v_time(qv_fitter,
                   nqubits: int,
                   original_ci: bool = False,
                   bootstrap_ci: bool = False,
                   fill_range: bool = False,
                   savename: Optional[str] = None):
    """Plot success as function of circuit index."""
          
    axis_font = 14

    ntrials = len(qv_fitter.heavy_output_counts)
    heavy_outputs = [
        qv_fitter.heavy_output_counts[f'qv_depth_{nqubits}_trial_{i}']/qv_fitter._circ_shots[f'qv_depth_{nqubits}_trial_{i}']
        for i in range(ntrials)
    ]
    ntrials = len(heavy_outputs)
    fig, ax = plt.subplots(figsize=(7,4))

    legend = []
    legend_name = []
    legend.append(
        ax.scatter(
            np.arange(ntrials), 
            heavy_outputs, 
            s=25, 
            color=ecolor[0], 
            alpha=0.4, 
            edgecolor='none'
        )
    )
    legend_name.append('Individual circuit')
    legend.append(
        ax.plot(
            np.arange(1, ntrials), 
            cumulative_average(heavy_outputs), 
            color=ecolor[0], 
            linewidth=2.5
        )[0]
    )
    legend_name.append('Average')

    if bootstrap_ci:
        b_lower = []
        b_upper = []
        for i in range(1, ntrials):
            lower_ci, upper_ci = bootstrap_bounds(
                qv_fitter, 
                reps=10000, 
                ntrials=i
            )
            b_lower.append(lower_ci)
            b_upper.append(upper_ci)
        if fill_range:
            legend.append(
                ax.plot(
                    np.arange(1, ntrials), 
                    np.array(b_upper), 
                    color=ecolor[2], 
                    linestyle='-', 
                    linewidth=1.5
                )[0]
            )
            ax.plot(
                np.arange(1, ntrials), 
                np.array(b_lower), 
                color=ecolor[2], 
                linestyle='-', 
                linewidth=1.5
            )
            ax.fill_between(
                np.arange(1, ntrials), 
                y1=np.array(b_upper), 
                y2=np.array(b_lower), 
                alpha=0.3, 
                color=ecolor[2]
            )
            legend_name.append('Bootstrap CI bound')
        else:
            legend.append(
                ax.plot(
                    np.arange(1, ntrials), 
                    np.array(b_lower), 
                    color=ecolor[2], 
                    linestyle='-', 
                    linewidth=2
                )[0]
            )
            legend_name.append('Bootstrap CI bound')

    if original_ci:
        o_lower = []
        o_upper = []
        for i in range(1, ntrials): 
            lower_ci, upper_ci = original_bounds(
                np.mean(heavy_outputs[:i]),
                i
            )
            o_lower.append(lower_ci)
            o_upper.append(upper_ci)
        if fill_range:
            legend.append(
                    ax.plot(
                        np.arange(1, ntrials), 
                        np.array(o_lower), 
                        color=ecolor[1], 
                        linestyle='-', 
                        linewidth=1.5
                    )[0]
                )
            ax.plot(
                np.arange(1, ntrials), 
                np.array(o_upper), 
                color=ecolor[1], 
                linestyle='-', 
                linewidth=1.5
            )
            ax.fill_between(
                np.arange(1, ntrials), 
                y1=np.array(o_upper), 
                y2=np.array(o_lower), 
                alpha=0.3, 
                color=ecolor[1],
                label='_nolegend_'
            )
            legend_name.append('Original CI bound')
        else:
            legend.append(
                ax.plot(
                    np.arange(1, ntrials), 
                    np.array(o_lower), 
                    color=ecolor[1], 
                    linestyle='-', 
                    linewidth=2
                )[0]
            )
            legend_name.append('Original CI bound')
    
    legend.append(
        ax.plot([0, ntrials], 
            [2/3]*2, 
            ':', 
            color='grey', 
            linewidth=2, 
        )[0]
    )
    legend_name.append('Passing threshold')

    ax.set_ylim(np.min(heavy_outputs)-0.02, np.max(heavy_outputs)+0.02)
    ax.set_xlim(0, ntrials)
    ax.set_ylabel('Heavy output frequency', fontsize=axis_font)
    ax.set_xlabel('Circuit index', fontsize=axis_font)
    ax.legend(
        legend,
        legend_name,
        loc='lower center',
        handlelength=3,
        fontsize=10
    )
    if savename:
        fig.savefig(savename + '.pdf', format='pdf')
        fig.savefig(savename + '.svg', format='svg')


def cumulative_average(sample):
    
    avg = [np.mean(sample[:i]) for i in range(1, len(sample))]
    
    return np.array(avg)