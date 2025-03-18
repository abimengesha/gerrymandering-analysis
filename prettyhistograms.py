# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:59:43 2024

@author: ozzma
"""

import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import time
import random
import numpy as np
import pandas as pd

graph = Graph.from_file("./IL/IL.shp")

ensembles = pd.read_csv("ensembles.csv")


elections = [
    Election("G20PRE", {"Democratic": "G20PRED", "Republican": "G20PRER"}),
    Election("G20USS", {"Democratic": "G20USSD", "Republican": "G20USSR"})
]

# Population updater, for computing how close to equality the district
# populations are. "TOTPOP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = Partition(
    graph,
    assignment="SSD",
    updaters=my_updaters
)


cutedge_ensemble = ensembles['cutedge_ensemble']

pres_demwin_ensemble = ensembles['pres_demwin_ensemble']
sen_demwin_ensemble = ensembles['sen_demwin_ensemble']

efficiency_gap_pres = ensembles['efficiency_gap_pres']
efficiency_gap_sen = ensembles['efficiency_gap_sen']

mean_median_diff_sen = ensembles['mean_median_diff_sen']
mean_median_diff_pres = ensembles['mean_median_diff_pres']


print(pres_demwin_ensemble[0])


# Plotting the histogram of the cut edges from the ensemble analysis
plt.figure()
plt.axvline(x =  len(initial_partition['cut_edges']), color = 'r', label = 'axvline - full height')
plt.hist(cutedge_ensemble, align='left')
plt.title('Cut Edges')
plt.show()

# Plotting the histogram of the Presidential elections won by Democrats
plt.figure()
plt.axvline(x = sen_demwin_ensemble[0], color = 'r', label = 'axvline - full height')
plt.hist(pres_demwin_ensemble, align='left')
plt.title('Presidential Elections Won by Democrats')
plt.show()

# Plotting the histogram of the Senate elections won by Democrats
plt.figure()
plt.axvline(x = pres_demwin_ensemble[0], color = 'r', label = 'axvline - full height')
plt.hist(sen_demwin_ensemble, align='left')
plt.title('Senate Elections Won by Democrats')
plt.show()

# Plotting the histogram for the mean-median difference in Presidential elections
plt.figure()
plt.axvline(x = mean_median_diff_pres[0], color = 'r', label = 'axvline - full height')
plt.hist(mean_median_diff_pres, align='left')
plt.title("Mean-Median Difference for Presidential Election")
plt.show()

# Plotting the histogram for the mean-median difference in Senate elections
plt.figure()
plt.axvline(x = mean_median_diff_sen[0], color = 'r', label = 'axvline - full height')
plt.hist(mean_median_diff_sen, align='left')
plt.title("Mean-Median Difference for Senate Election")
plt.show()

# Plotting the histogram for the efficiency gap in Presidential elections
plt.figure()
plt.axvline(x = efficiency_gap_pres[0], color = 'r', label = 'axvline - full height')
plt.hist(efficiency_gap_pres, align='left')
plt.title("Efficiency Gap for Presidential Election")
plt.show()


# Plotting the histogram for the efficiency gap in Senate elections
plt.figure()
plt.axvline(x = efficiency_gap_sen[0], color = 'r', label = 'axvline - full height')
plt.hist(efficiency_gap_sen, align='left')
plt.title("Efficiency Gap for Senate Election")
plt.show()
