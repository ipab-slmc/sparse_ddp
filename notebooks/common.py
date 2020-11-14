from __future__ import print_function, division

import pyexotica as exo
import exotica_ddp_solver_py
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
# from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import matplotlib
import sys


def load_problem(path, problem_name, solver_name, solver_params=None, problem_params=None):
    # Load problem
    solver_config, problem_config = exo.Initializers.load_xml_full(
        path, solver_name=solver_name, problem_name=problem_name
    )

    if problem_params is not None:
        for key, value in problem_params.items():
            problem_config[1][key] = value
    problem = exo.Setup.create_problem(problem_config)

    if solver_params is not None:
        for key, value in solver_params.items():
            solver_config[1][key] = value
    solver = exo.Setup.create_solver(solver_config)
    solver.specify_problem(problem)

    del solver_config, problem_config

    return problem, solver


def compute_work(X, U):
    # Compute work done
    # tau * delta theta
    work = 0
    # this gives me the delta q between two subsequent timesteps
    xdiff = np.diff(X, axis=1)
    for t in range(xdiff.shape[1]):  # for each timestep
        # print(t, "xdiff", xdiff[:,t], "u", U[:,t])
        for control_dim in range(U.shape[0]):  # for each control dimension
            # abs because every control input is expended
            work += abs(xdiff[control_dim, t]) * abs(U[control_dim, t])
    return work


def find_result_with_most_zero_controls(results, vel_threshold=1e-2):
    # assert results['X'][0].shape[1] == 4 # ONLY CARTPOLE SUPPORTED RIGHT NOW !!!
    best = None
    best_zero = -1

    for res in results:
        # pick solution with 0 end velocities
        final_pos = np.abs(res['X'][:, -1][:2])   # HARDCODED!
        target_pos = np.array([0, np.pi])
        final_vels = np.abs(res['X'][:, -1][-2:]) # HARDCODED!
        num_zero = np.sum(np.isclose(res['solution'], 0, atol=1e-2))

        # print(">>>", res['loss_type'], "alpha=", res['reg_rate'], "lambda=", res['reg_strength'], "final control cost=", res['control_cost_evolution'][-1], "final cost=", res['cost_evolution']['values'][-1], "num_zero=", num_zero, "vel OK?", np.all(final_vels < vel_threshold))

        if np.all(final_vels < vel_threshold) and np.allclose(final_pos, target_pos, atol=1e-2) and num_zero > best_zero:
            best = res
            best_zero = num_zero
            best_cost = res['cost_evolution']['values'][-1]
            print(">>>", res['loss_type'], '(ws)' if res['warmstart'] else '', "alpha=", res['reg_rate'], "lambda=", res['reg_strength'], "new best (zero):", best_cost, best_zero)
    if best is None:
        i = 0
        for res in results:
            final_vels = np.abs(res['X'][:, -1][-2:]) # HARDCODED!
            num_zero = np.sum(np.isclose(res['solution'], 0, atol=1e-2))
            print(i, np.max(final_vels), num_zero)
            i += 1

        raise RuntimeWarning('No solution satisfies velocity threshold!')
        raise
    return best, best_zero

def find_result_with_lowest_cost(results, vel_threshold=1e-2):
    # print(results['X'][0,-1].shape)
    # assert results['X'][0].shape[1] == 4 # ONLY CARTPOLE SUPPORTED RIGHT NOW !!!
    best = None
    best_zero = -1
    best_cost = 1e100

    costs = []
    for res in results:
        # pick solution with 0 end velocities
        vels = np.abs(res['X'][:, -1][-2:]) # HARDCODED!
        num_zero = np.sum(np.isclose(res['solution'], 0, atol=1e-2))

        costs.append(res['cost_evolution']['values'][-1])

        if np.all(vels < vel_threshold) and res['cost_evolution']['values'][-1] < best_cost:
            best = res
            best_zero = num_zero
            best_cost = res['cost_evolution']['values'][-1]
            print("new best:", best_cost, best_zero, res['iterations'], "reg=", res['reg_strength'], res['reg_rate'])
        else:
            if np.all(vels < vel_threshold): print("cost not good enough:", res['cost_evolution']['values'][-1], res['iterations'], "reg=", res['reg_strength'], res['reg_rate'])
            else: print("velocity too max", np.max(vels), res['iterations'], "reg=", res['reg_strength'], res['reg_rate'])
    if best is None:
        i = 0
        for res in results:
            vels = np.abs(res['X'][:, -1][-2:]) # HARDCODED!
            num_zero = np.sum(np.isclose(res['solution'], 0, atol=1e-2))
            print(i, np.max(vels), num_zero)
            i += 1

        raise RuntimeWarning('No solution satisfies velocity threshold!')
        raise
    costs.sort()
    print("All costs", costs)
    return best, best_zero


def compute_number_of_zero_controls(solution, zero_atol=1e-2):
    # / solution.shape[0]
    return np.sum(np.isclose(solution, 0, atol=zero_atol))


def grid_search(config, problem_name, solver_name, reg_rates, reg_strengths, qf_rate, loss_type, warmstart=None, warmstart_runtime=0,
                l2_rate='0', include_cost_evolution=True):
    if loss_type not in ['SmoothL1', 'L2', 'Huber', 'NormalizedHuber']:
        raise ValueError('Loss not supported!')

    results = []
    problem = None

    pbar = tqdm(total=len(reg_rates) * len(reg_strengths))

    for reg_rate in reg_rates:
        for reg_strength in reg_strengths:
            problem_params = {
                'Qf_rate': str(qf_rate),
                'LossType': loss_type,

                'L1Rate': str(reg_rate),
                'HuberRate': str(reg_rate),
                'R_rate': str(l2_rate),

                'ControlCostWeight': str(reg_strength)
            }

            problem, solver = load_problem(
                'exotica_configs/{0}.xml'.format(config),
                problem_name, solver_name,
                problem_params=problem_params
            )

            if warmstart is not None:
                for i, u in enumerate(warmstart):
                    problem.update(u, i)

            # it doesn't work without this ?!?
            # time.sleep(0.1)

            start_time = time.time()
            solution = solver.solve()
            end_time = time.time()

            work = compute_work(problem.X, problem.U)

            result = {
                'loss_type': loss_type,
                'qf_rate': qf_rate,
                'reg_rate': reg_rate,
                'solution': solution,
                'final_cost': problem.get_state_cost(-1) + problem.get_control_cost(-1),
                'work': work,
                'regularization_evolution': solver.regularization_evolution,
                'steplength_evolution': solver.steplength_evolution,
                'runtime': end_time - start_time + warmstart_runtime,
                'final_state': problem.X[:, -1],
                'final_state_cost': problem.get_state_cost(-1),
                'X': problem.X,
                'reg_strength': reg_strength,
                'warmstart': warmstart is not None,
                'termination_criterion': problem.termination_criterion,
                'iterations': len(problem.get_cost_evolution()[1])
            }

            if include_cost_evolution:
                result['cost_evolution'] = {
                    'times': problem.get_cost_evolution()[0],
                    'values': problem.get_cost_evolution()[1]
                }
                result['control_cost_evolution'] = solver.control_cost_evolution

            results.append(result)

            # print(sys.getsizeof(results))
            # time.sleep(1)

            pbar.update()

    times = [0]
    for i in range(problem.T - 1):
        times.append(times[-1] + problem.tau)
    times = times[1:]

    del problem
    del solver

    return results, times

def create_table_of_results(problem_name, zero_atol, *results):
    all_data = {
        'Problem': [],
        'Loss': [],
        'Reg. Rate': [],
        'Reg. Strength': [],
        'Cost': [],
        'Final State Cost': [],
        '# Zero Controls': [],
        'Runtime': []
    }

    for res in results:
        for r in res:
            all_data['Problem'].append(problem_name)
            if r['warmstart']:
                all_data['Loss'].append(r['loss_type'] + ' (warmstart)')
            else:
                all_data['Loss'].append(r['loss_type'])

            if type(r['reg_rate']) is str:
                all_data['Reg. Rate'].append(r['reg_rate'].split()[0])
            else:
                all_data['Reg. Rate'].append(r['reg_rate'])

            all_data['Reg. Strength'].append(r['reg_strength'])

            # all_data['Cost'].append(r['cost_evolution']['values'][-1])
            all_data['Cost'].append(r['final_cost'])
            # all_data['Runtime'].append(r['cost_evolution']['times'][-1])
            all_data['Runtime'].append(r['runtime'])

            all_data['Final State Cost'].append(r['final_state_cost'])
            all_data['# Zero Controls'].append(
                np.sum(np.isclose(r['solution'], 0, atol=zero_atol)))

    pd_all = pd.DataFrame(all_data, columns=['Problem', 'Loss', 'Reg. Rate', 'Reg. Strength', 'Cost', 'Final State Cost',
                                             '# Zero Controls', 'Runtime'])
    return pd_all
