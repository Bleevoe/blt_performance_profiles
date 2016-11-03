"""
Solves optimal control problems using various elimination schemes of JModelica.org.

The script first has a setup section where some benchmark settings easily can be changed, such as number of instances to
consider for each problem. Changing other things, such as which schemes to use, is possible but requires more effort,
as some parts are hardcoded. A stats file is generated regularly during execution (which can last for hours). These
stats files are then used by the performance_profile.py script to plot the corresponding performance profile and by the
process_stats.py script to print some numbers.

Running all schemes on all six problems simultaneously requires a lot of RAM (more than 32 GB). It is thus recommended
to only run a single problem at a time. Some parts of the scripts support multiple problems simultaneously, but other do
not. It is however possible (and recommended) to run multiple Python instances simultaneously, each with its own
problem.

Note that the scripts refer to the scheme with tearing and without sparsity preservation as scheme 3, instead of 2, and
vice versa. The numbering in the performance profile legend are however consistent with the publications.
"""

################################################### Benchmark setup ####################################################
n_runs = 10 # Number of instances per problem
old_stats_file = None # Set an old stats file to continue appending results to it
problems = ["dist"] # Possible values: car, ccpp, double_pendulum, fourbar1, dist
########################################################################################################################

try:
    import pyjmi
    import pymodelica
    import pyfmi
except ImportError:
    raise ImportError('Unable to find JModelica.org installation.')
from pyjmi.symbolic_elimination import BLTOptimizationProblem, EliminationOptions
from pyjmi import transfer_optimization_problem, get_files_path
from pyjmi.optimization.casadi_collocation import BlockingFactors
from pymodelica import compile_fmu
from pyfmi import load_fmu
import os
from pyjmi.common.io import ResultDymolaTextual
from pyjmi.optimization.casadi_collocation import LocalDAECollocationAlgResult
import time
import numpy as np
import scipy.io as sio
from itertools import izip
import pickle

# Specify schemes for each problem
schemes = {}
if "car" in problems:
    schemes["car"] = ["0", "1", "2.05"]
if "ccpp" in problems:
    schemes["ccpp"] = ["0", "1", "2.05", "3", "4.05"]
if "double_pendulum" in problems:
    schemes['double_pendulum'] = ["0", "1", "2.05", "3", "4.05", "4.10"]
if "fourbar1" in problems:
    schemes['fourbar1'] = ["0", "1", "2.05", "2.10", "2.20", "3",
                           "4.05", "4.10", "4.20", "4.30", "4.40"]
if "dist" in problems: # Exclude scheme 0
    schemes['dist'] = ["1", "2.05", "2.10", "2.20", "2.30", "2.40",
                        "3", "4.05", "4.10", "4.20", "4.30", "4.40"]

# Load existing stats file
if old_stats_file is None:
    stats = {}
    for problem in problems:
        stats[problem] = dict([(scheme, []) for scheme in schemes[problem]])
else:
    stats = pickle.load(open(old_stats_file, "rb"))
    for problem in problems:
        if problem in stats:
            for scheme in stats[problem].keys():
                if scheme not in schemes[problem]:
                    del stats[problem][scheme]
        else:
            stats[problem] = dict([(scheme, []) for scheme in schemes[problem]])

### For each problem, set options and compile ###
ops = {}
solvers = {}
n_algs = {}
std_dev = {}
init_res = {}
for problem in problems:
    opt_opts = {}
    opt_opts['IPOPT_options'] = {}
    opt_opts['IPOPT_options']['acceptable_iter'] = 10000
    opt_opts['IPOPT_options']['acceptable_tol'] = 1e-12
    opt_opts['IPOPT_options']['acceptable_constr_viol_tol'] = 1e-12
    opt_opts['IPOPT_options']['acceptable_dual_inf_tol'] = 1e-12
    opt_opts['IPOPT_options']['acceptable_compl_inf_tol'] = 1e-12
    opt_opts['IPOPT_options']['linear_solver'] = "ma57"
    opt_opts['IPOPT_options']['ma57_pivtol'] = 1e-4
    opt_opts['IPOPT_options']['ma57_automatic_scaling'] = "yes"
    opt_opts['IPOPT_options']['mu_strategy'] = "adaptive"
    if problem == "car":
        std_dev[problem] = 0.1
        caus_opts = EliminationOptions()
        caus_opts['uneliminable'] = ['car.Fxf', 'car.Fxr', 'car.Fyf', 'car.Fyr']
        class_name = "Turn"
        file_paths = os.path.join(get_files_path(), "vehicle_turn.mop")
        init_res[problem] = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('sols/car_sol.txt'))
        opt_opts['init_traj'] = init_res[problem]
        opt_opts['nominal_traj'] = init_res[problem]
        opt_opts['IPOPT_options']['max_cpu_time'] = 30
        opt_opts['n_e'] = 60

        # Set blocking factors
        factors = {'delta_u': opt_opts['n_e'] / 2 * [2],
                   'Twf_u': opt_opts['n_e'] / 4 * [4],
                   'Twr_u': opt_opts['n_e'] / 4 * [4]}
        rad2deg = 180. / (2*np.pi)
        du_bounds = {'delta_u': 2. / rad2deg}
        bf = BlockingFactors(factors, du_bounds=du_bounds)
        opt_opts['blocking_factors'] = bf

        # Set up optimization problems for each scheme
        compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}
        ops[problem] = {}
        solvers[problem] = {}
        n_algs[problem] = {}
        dns_tol = 5
        
        # Scheme 0
        scheme = "0"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 1
        scheme = "1"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.05
        scheme = "2.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = dns_tol
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
    elif problem == "ccpp":
        std_dev[problem] = 0.3
        caus_opts = EliminationOptions()
        caus_opts['uneliminable'] = ['plant.sigma']
        caus_opts['tear_vars'] = ['plant.turbineShaft.T__3']
        caus_opts['tear_res'] = [123]
        class_name = "CombinedCycleStartup.Startup6"
        file_paths = (os.path.join(get_files_path(), "CombinedCycle.mo"),
                      os.path.join(get_files_path(), "CombinedCycleStartup.mop"))
        init_res[problem] = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('sols/ccpp_sol.txt'))
        opt_opts['init_traj'] = init_res[problem]
        opt_opts['nominal_traj'] = init_res[problem]
        opt_opts['IPOPT_options']['max_cpu_time'] = 40
        opt_opts['n_e'] = 40
        opt_opts['n_cp'] = 4
        compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}

        # Set up FMU to check initial state feasibility
        fmu = load_fmu(compile_fmu("CombinedCycleStartup.Startup6Verification", file_paths,
                                   separate_process=True, compiler_options=compiler_opts))

        # Set up optimization problems for each scheme
        ops[problem] = {}
        solvers[problem] = {}
        n_algs[problem] = {}
        
        # Scheme 0
        scheme = "0"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 1
        scheme = "1"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.05
        scheme = "2.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 3
        scheme = "3"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.05
        scheme = "4.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
    elif problem == "double_pendulum":
        std_dev[problem] = 0.3
        caus_opts = EliminationOptions()
        caus_opts['tear_vars'] = ['der(pendulum.boxBody1.body.w_a[3])', 'der(pendulum.boxBody2.body.w_a[3])']
        caus_opts['tear_res'] = [43, 44]
        init_res[problem] = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('sols/dbl_pend_sol.txt'))
        class_name = "Opt"
        file_paths = (os.path.join(get_files_path(), "DoublePendulum.mo"),
                      os.path.join(get_files_path(), "DoublePendulum.mop"))
        compiler_opts = {'generate_html_diagnostics': False, 'inline_functions': 'all', 'dynamic_states': False,
                         'state_initial_equations': False, 'equation_sorting': True, 'automatic_tearing': True}
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
        opt_opts['init_traj'] = init_res[problem]
        opt_opts['nominal_traj'] = init_res[problem]
        opt_opts['IPOPT_options']['max_cpu_time'] = 50
        opt_opts['n_e'] = 100

        # Set up optimization problems for each scheme
        ops[problem] = {}
        solvers[problem] = {}
        n_algs[problem] = {}

        # Scheme 0
        scheme = "0"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 1
        scheme = "1"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.05
        scheme = "2.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 3
        scheme = "3"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 4.05
        scheme = "4.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 4.10
        scheme = "4.10"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
    elif problem == "fourbar1":
        std_dev[problem] = 0.03
        caus_opts = EliminationOptions()
        uneliminable = ['fourbar1.j2.s']
        uneliminable += ['fourbar1.j3.frame_a.f[1]', 'fourbar1.b0.frame_a.f[3]']
        caus_opts['uneliminable'] = uneliminable
        caus_opts['tear_vars'] = [
                'fourbar1.j4.phi', 'fourbar1.j3.phi', 'fourbar1.rev.phi', 'fourbar1.rev1.phi', 'fourbar1.j5.phi',
                'der(fourbar1.rev.phi)', 'der(fourbar1.rev1.phi)', 'temp_2962', 'der(fourbar1.j5.phi)', 'temp_2943',
                'der(fourbar1.rev.phi,2)', 'der(fourbar1.b3.body.w_a[3])', 'der(fourbar1.j4.phi,2)',
                        'der(fourbar1.j5.phi,2)', 'temp_3160', 'temp_3087',
                        'fourbar1.b3.frame_a.t[1]', 'fourbar1.b3.frame_a.f[1]']
        caus_opts['tear_res'] = [160, 161, 125, 162, 124,
                                 370, 356, 306, 258, 221,
                                 79, 398, 383, 411, 357, 355, 257, 259]
        init_res[problem] = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('sols/fourbar1_sol.txt'))
        file_paths = (os.path.join(get_files_path(), "Fourbar1.mo"),
                      os.path.join(get_files_path(), "Fourbar1.mop"))
        class_name = "Opt"
        compiler_opts = {'generate_html_diagnostics': True, 'inline_functions': 'all', 'dynamic_states': False,
                         'state_initial_equations': False}
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
        op.set('finalTime', 1.0)
        opt_opts['init_traj'] = init_res[problem]
        opt_opts['nominal_traj'] = init_res[problem]
        opt_opts['IPOPT_options']['max_cpu_time'] = 30
        opt_opts['n_e'] = 60

        # Set up optimization problems for each scheme
        ops[problem] = {}
        solvers[problem] = {}
        n_algs[problem] = {}

        # Scheme 0
        scheme = "0"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 1
        scheme = "1"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.05
        scheme = "2.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.10
        scheme = "2.10"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 2.20
        scheme = "2.20"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 20
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 3
        scheme = "3"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.05
        scheme = "4.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 4.10
        scheme = "4.10"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 4.20
        scheme = "4.20"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 20
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        
        # Scheme 4.30
        scheme = "4.30"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 30
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.40
        scheme = "4.40"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 40
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
    elif problem == "dist":
        std_dev[problem] = 0.3
        caus_opts = EliminationOptions()
        caus_opts['uneliminable'] = ['Dist', 'Bott']
        caus_opts['tear_vars'] = (['Temp[%d]' % i for i in range(1, 43)] + 
                                  ['V[%d]' % i for i in range(2, 42)] + ['L[41]'] +
                                  ['der(xA[%d])' % i for i in range(2, 43)])
        caus_opts['tear_res'] = range(1083, 1125) + range(1042, 1083) + range(673, 714)
        class_name = "JMExamples_opt.Distillation4_Opt"
        file_paths = (os.path.join(get_files_path(), "JMExamples.mo"),
                      os.path.join(get_files_path(), "JMExamples_opt.mop"))
        init_res[problem] = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('sols/dist_sol.txt'))
        opt_opts['init_traj'] = init_res[problem]
        opt_opts['nominal_traj'] = init_res[problem]
        opt_opts['IPOPT_options']['max_cpu_time'] = 40

        # Local
        opt_opts['n_e'] = 20

        # Global
        #~ opt_opts['n_e'] = 1
        #~ opt_opts['n_cp'] = 25
        
        compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}

        # Set up optimization problems for each scheme
        ops[problem] = {}
        solvers[problem] = {}
        n_algs[problem] = {}
        
        # Scheme 0
        scheme = "0"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 1
        scheme = "1"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.05
        scheme = "2.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.10
        scheme = "2.10"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.20
        scheme = "2.20"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 20
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.30
        scheme = "2.30"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 30
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 2.40
        scheme = "2.40"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 40
            caus_opts['tearing'] = False
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 3
        scheme = "3"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.05
        scheme = "4.05"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.10
        scheme = "4.10"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.20
        scheme = "4.20"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 20
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.30
        scheme = "4.30"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 30
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

        # Scheme 4.40
        scheme = "4.40"
        if scheme in schemes[problem]:
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 40
            caus_opts['tearing'] = True
            op = BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
    else:
        raise ValueError("Unknown problem %s." % problem)

    # Print algebraics
    print("Algebraic variables:")
    for scheme in sorted(n_algs[problem].keys()):
        print('%s: %d' % (scheme, n_algs[problem][scheme]))
    print("\n")

### Execute ###
for problem in problems:
    # Perturb initial state
    np.random.seed(1)
    op0 = ops[problem].values()[0] # Get arbitrary OP to compute min and max
    x_vars = op0.getVariables(op.DIFFERENTIATED)
    x_names = [x_var.getName() for x_var in x_vars]
    x0 = [init_res[problem].initial(var.getName()) for var in x_vars]
    [x_min, x_max] = zip(*[(op0.get_attr(var, "min"), op0.get_attr(var, "max")) for var in x_vars])
    if problem == "dist":
        x_min = tuple(42*[0.])
        x_max = tuple(42*[1.])
    x0_pert_min = []
    x0_pert_max = []

    # Move perturbations inside of bounds
    for (var_nom, var_min, var_max) in izip(x0, x_min, x_max):
        x0_pert_min.append(var_nom - 0.9*(var_nom-var_min))
        x0_pert_max.append(var_nom + 0.9*(var_max-var_nom))

    # Solve
    for i in xrange(n_runs):
        x0_pert = x0
        feasible = False
        while not feasible:
            x0_pert = np.random.normal(1, std_dev[problem], len(x0)) * x0
            x0_pert_proj = [min(max(val, val_min), val_max)
                            for (val, val_min, val_max) in izip(x0_pert, x0_pert_min, x0_pert_max)]
            if problem == "car":
                X = x0_pert_proj[3]
                if X > 35.:
                    feasible = True
                else:
                    feasible = False
            elif problem == "ccpp":
                fmu.reset()
                fmu.set(['_start_' + name for name in  x_names], x0_pert_proj)
                try:
                    fmu.initialize()
                except:
                    feasible = False
                else:
                    sigma = np.array(fmu.get(['plant.sigma']))
                    if all(sigma < 0.9*2.6e8):
                        feasible = True
                    else:
                        feasible = False
            else:
                feasible = True
        for scheme in schemes[problem]:
            if i >= len(stats[problem][scheme]):
                print('%s, scheme %s: %d/%d' % (problem, scheme, i+1, n_runs))
                solver = solvers[problem][scheme]
                if problem == "fourbar1":
                    solver.set('phi_start', x0_pert_proj[0])
                    solver.set('w_start', x0_pert_proj[1])
                elif problem == "double_pendulum":
                    solver.set('phi1_start', x0_pert_proj[0])
                    solver.set('w1_start', x0_pert_proj[1])
                    solver.set('phi2_start', x0_pert_proj[0])
                    solver.set('w2_start', x0_pert_proj[1])
                else:
                    solver.set(['_start_' + var.getName() for var in x_vars], x0_pert_proj)
                res = solver.optimize()
                stats[problem][scheme].append(res.get_solver_statistics())
        if (i+1) >= len(stats[problem][scheme]) and ((i+1) % 50 == 0 or (i+1) in [10, 20, 30, 40]):
            file_name = 'stats/stats_%s_%d_%d' % (problem, 100*std_dev[problem], int(time.time()))
            pickle.dump(stats, open(file_name, "wb"))
    file_name = 'stats/stats_%s_%d_%d' % (problem, 100*std_dev[problem], int(time.time()))
    pickle.dump(stats, open(file_name, "wb"))
print(file_name)
