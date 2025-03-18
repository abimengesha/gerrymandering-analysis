# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:38:52 2024

@author: Austin Asher
"""
import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from gerrychain.updaters import Tally, cut_edges
from functools import partial
import pandas as pd

ensembles = pd.read_csv("ensembles.csv")

graph = Graph.from_file("./IL/IL.shp")

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

tot_pop = sum([graph.nodes()[v]['TOTPOP'] for v in graph.nodes()])
num_dist = 59
ideal_pop = tot_pop/num_dist
pop_tolerance = .1

#random walk parameters
proposal = partial(recom, ## how you choose a next districting plan
                      pop_col = "TOTPOP", ## What data describes population? 
                      pop_target = ideal_pop, ## What the target/ideal population is for each district 
                                              ## (we calculated ideal pop above)
                      epsilon = pop_tolerance,  ## how far from ideal population you can deviate
                                              ## (we set pop_tolerance above)
                      node_repeats = 1 ## number of times to repeat bipartition.  Can increase if you get a BipartitionWarning
                      )


def cut_edges_length(p):
  return len(p["cut_edges"])

compactness_bound = constraints.UpperBound(
  cut_edges_length,
  2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.1)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000
)

# This might take a few minutes.

data = pd.DataFrame(
    sorted(partition["G20USS"].percents("Democratic"))
    for partition in chain.with_progress_bar()
)

fig, ax = plt.subplots(figsize=(15, 6))

# Draw 50% line
ax.axhline(0.5, color="#cccccc")

# Draw boxplot
data.boxplot(ax=ax, positions=range(len(data.columns)), showfliers = False)

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
plt.plot(data.iloc[0], "ro")

# Annotate
ax.set_title("Comparing the 2021 plan to an ensemble")
ax.set_ylabel("Democratic vote % (Senate 2020)")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

plt.show()

