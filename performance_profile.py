"""
Create performance profile for stats files generated with benchmark.py. It is recommended to have one stats file per
problem.

Some schemes are identical for some problems. This information is hardcoded manually for each problem and scheme.
Consequently, some extra effort is required to try new values of the density tolerance.

The stats files used to create the published performance profiles are used by default, albeit without the HRSG
results.
"""

######################################################## Setup #########################################################
# dict with problem: stats_file
stats_files = {'car': 'stats/stats_car_10',
               'ccpp': 'stats/stats_ccpp_30', 
               'double_pendulum': 'stats/stats_double_pendulum_30', 
               'fourbar1': 'stats/stats_fourbar1_3', 
               'dist': 'stats/stats_dist_30'} 
n_tau = 100 # Number of sample points for tau
########################################################################################################################

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Scheme aesthetics
schemes =       ["0" , "1"  ,  "3", "2.40"  , "2.30"  , "2.20"  , "2.10"  , "2.05"  ,
                 "4.40"   , "4.30"  , "4.20"  , "4.10"  , "4.05"]
scheme_labels = ["0" , "1"  ,  "2", "3_{40}", "3_{30}", "3_{20}", "3_{10}", "3_{05}",
                 "4_{40}" , "4_{30}", "4_{20}", "4_{10}", "4_{5}"]
schm_clr_idxs = [0,        0,    0,      1  ,        2,        3,        4,        5,
                 1   ,        2,        3,        4,        5]
scheme_styles = ["-" , "--" , "-.", "--"    , "--"    , "--"    , "--"    , "--",
                 "-."     , "-."    , "-."    , "-."    , "-."]
scheme_labels = map(lambda s: "$" + s + "$", scheme_labels)

# Change this to only consider certain schemes
scheme_idxs = range(len(schemes))
if len(scheme_idxs) != len(schemes):
    pass
    #~ scheme_styles = ['-', '-', '-', '--', '--']
    #~ schm_clr_idxs = [0, 2, 5, 2, 5]
    #~ scheme_styles = ['-', '--']
    #~ schm_clr_idxs = [0, 5]

# Compute scheme colors
cNorm = matplotlib.colors.Normalize(vmin=0, vmax=6)
scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap='nipy_spectral')
scheme_colors = map(scalarMap.to_rgba, schm_clr_idxs)
if len(scheme_idxs) == len(schemes):
    scheme_colors[6] = scheme_colors[11] = (0.95, 0.6, 0.0, 1.0)

# Load stats and define scheme equalities
statses = dict(zip(stats_files, [pickle.load(open(stats_files[problem], "rb")).values()[0] for problem in stats_files]))
if "car" in stats_files:
    statses["car"]["2.10"] = statses["car"]["2.05"]
    statses["car"]["2.20"] = statses["car"]["1"]
    statses["car"]["2.30"] = statses["car"]["1"]
    statses["car"]["2.40"] = statses["car"]["1"]
    statses["car"]["3"] = statses["car"]["1"]
    statses["car"]["4.05"] = statses["car"]["2.05"]
    statses["car"]["4.10"] = statses["car"]["2.10"]
    statses["car"]["4.20"] = statses["car"]["2.20"]
    statses["car"]["4.30"] = statses["car"]["2.30"]
    statses["car"]["4.40"] = statses["car"]["2.40"]
if "ccpp" in stats_files:
    statses["ccpp"]["2.10"] = statses["ccpp"]["1"]
    statses["ccpp"]["2.20"] = statses["ccpp"]["1"]
    statses["ccpp"]["2.30"] = statses["ccpp"]["1"]
    statses["ccpp"]["2.40"] = statses["ccpp"]["1"]
    statses["ccpp"]["4.10"] = statses["ccpp"]["3"]
    statses["ccpp"]["4.20"] = statses["ccpp"]["3"]
    statses["ccpp"]["4.30"] = statses["ccpp"]["3"]
    statses["ccpp"]["4.40"] = statses["ccpp"]["3"]
if "fourbar1" in stats_files:
    statses["fourbar1"]["2.30"] = statses["fourbar1"]["1"]
    statses["fourbar1"]["2.40"] = statses["fourbar1"]["1"]
if "double_pendulum" in stats_files:
    statses["double_pendulum"]["2.10"] = statses["double_pendulum"]["1"]
    statses["double_pendulum"]["2.20"] = statses["double_pendulum"]["1"]
    statses["double_pendulum"]["2.30"] = statses["double_pendulum"]["1"]
    statses["double_pendulum"]["2.40"] = statses["double_pendulum"]["1"]
    statses["double_pendulum"]["4.20"] = statses["double_pendulum"]["3"]
    statses["double_pendulum"]["4.30"] = statses["double_pendulum"]["3"]
    statses["double_pendulum"]["4.40"] = statses["double_pendulum"]["3"]
schemes = [schemes[i] for i in scheme_idxs]
scheme_labels = [scheme_labels[i] for i in scheme_idxs]

# Compute normalized solution times
r = {}
for scheme in schemes:
    r[scheme] = []
n_p = 0.
for problem in statses:
    stats = statses[problem]
    n_runs = len(stats.values()[0])
    for i in xrange(n_runs):
        times = [stats[scheme][i][3] for scheme in schemes if stats[scheme][i][0] == "Solve_Succeeded"]
        if len(times) > 0:
            n_p += 1
            t_min = np.min(times)
            for scheme in schemes:
                if stats[scheme][i][0] == "Solve_Succeeded":
                    time = stats[scheme][i][3]
                    r[scheme].append(time / t_min)
                else:
                    r[scheme].append(np.inf)

# Plot
def rho(r, s, tau):
    return sum(r[s] <= tau)/n_p
plt.close(1)
plt.figure(1, figsize=(12, 9))
plt.rcParams.update(
    {'legend.fontsize': 24,
     'axes.labelsize': 28,
     'xtick.labelsize': 24,
     'ytick.labelsize': 24})
taus = np.logspace(0, 2, n_tau)
for (scheme, color, style) in zip(schemes, scheme_colors, scheme_styles):
    plt.semilogx(taus, [rho(r, scheme, tau) for tau in taus], color=color, linestyle=style, lw=2)
plt.legend(scheme_labels, loc='lower right')
plt.xlabel('$\\tau$')
plt.ylabel('$\\rho(\\tau)$')
plt.show()
