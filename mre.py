import numpy as np
import pandas as pd
import warnings
import logging
import time
import random

from pgmpy.inference import VariableElimination
from EDAspy.optimization import EBNA, UMDAcat

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax

import inspyred
from inspyred.ec import terminators

__all__ = [' UMDAcat_mre','UMDAcat2_mre','ebna_mre','nsga2_mre','tabu_mre','brute_force_mre_pruning',
           'ga_mre','es_mre','dea_mre','pso_mre','hierarchical_beam_search']


_states = None
_evidence = None
_model = None
_inference = None
_possible_states = None     # Diccionario con 'Nombre nodo': 'numero de estados' 
_segmented_solutions = {}
_more_targets = 0

warnings.filterwarnings("ignore")
logger = logging.getLogger("pgmpy")
logger.setLevel(logging.ERROR)


def calculate_gbf(model, explanation, evidence):
    """
    Compute gbf of explanation given the evidence

    Parameters
    ----------
    model :
        bayesian network model for inference
    explanation : dict
        dict with the explanation nodes and values
    evidence : dict
        dict containing the evidence nodes and values

    Returns
    --------
    gbf value
    
    """
    global _inference

    phi_query1 = _inference.query(explanation,evidence=evidence)
    phi_query2 = _inference.query(explanation)
    p1 = phi_query1.get_value(**explanation)
    p2 = phi_query2.get_value(**explanation)

    if (p2*(1-p1)) == 0:
        return 0

    gbf = (p1*(1-p2))/(p2*(1-p1))
    return gbf

def brute_force_mre(model,evidence,target=None):
    """
    Computes mre by brute force approach, although it repeats calculations due to not 
    taking into account the combinations already tested

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes

    Returns
    ----------
    - a tuple with best gbf value and dict with the solution
    - elapsed time
    """
    global _inference
    st = time.time()
    _inference = VariableElimination(model)
    states = list(model.states.keys())
    states = [x for x in states if x not in evidence.keys()]
    if target!=None:
        states = target
    
    values = []
    for s in states:
        values.append(model.states[s])
    
    gbf = (0,{})

    for i,s in enumerate(states):
        for v in range(0,len(values[i])):
            re = _calculate_mre({s:values[i][v]},0,model,evidence,states,values)
            if re[0] > gbf[0]:
                gbf = re

    et = time.time()
    elapsed_time = et - st

    return (gbf,elapsed_time)

def _calculate_mre(sol_actual,state_index, model, evidence, states, values):
    gbf = calculate_gbf(model,sol_actual,evidence)

    state_index+=1

    if state_index >= len(states):
        return (gbf,sol_actual)

    max_gbf= (gbf, sol_actual)

    re = _calculate_mre(sol_actual,state_index,model,evidence,states,values)
    if re[0] > max_gbf[0]:
        max_gbf = re
    for v in range(0,len(values[state_index])):
        new_sol = sol_actual.copy()
        new_sol[states[state_index]] = values[state_index][v]
        re = _calculate_mre(new_sol,state_index,model,evidence,states,values)
        if re[0] > max_gbf[0]:
            max_gbf = re

    return max_gbf

def brute_force_mre_pruning(model,evidence,target=None):
    """
    Computes mre by brute force approach, pruning visited states

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes

    Returns
    ----------
    - a tuple with best gbf value and dict with the solution
    - elapsed time
    """
    global _inference
    st = time.time()
    _inference = VariableElimination(model)
    states = list(model.states.keys())
    states = [x for x in states if x not in evidence.keys()]
    if target!=None:
        states = target
    
    values = []
    for s in states:
        values.append(model.states[s])
    
    gbf = (0,{})

    visited_states = []

    for i,s in enumerate(states):
        for v in range(0,len(values[i])):
            re = _calculate_mre_pruning({s:values[i][v]},0,model,evidence,states,values,visited_states)
            if re[0] > gbf[0]:
                gbf = re
        visited_states.append(s)

    et = time.time()
    elapsed_time = et - st

    return (gbf,elapsed_time)


def _calculate_mre_pruning(sol_actual,state_index, model, evidence, states, values,visited_states):
    gbf = calculate_gbf(model,sol_actual,evidence)

    state_index+=1

    if state_index >= len(states):
        return (gbf,sol_actual)

    max_gbf= (gbf, sol_actual)

    re = _calculate_mre_pruning(sol_actual,state_index,model,evidence,states,values,visited_states)
    if re[0] > max_gbf[0]:
        max_gbf = re
    if states[state_index] in visited_states:
        return max_gbf
    for v in range(0,len(values[state_index])):
        new_sol = sol_actual.copy()
        new_sol[states[state_index]] = values[state_index][v]
        re = _calculate_mre_pruning(new_sol,state_index,model,evidence,states,values,visited_states)
        if re[0] > max_gbf[0]:
            max_gbf = re

    return max_gbf

def _categorical_cost_function(solution):
    global _model, _evidence, _states
    # print(solution)
    di = {}
    for index, value in enumerate(_states):
        if solution[index] != 'None' and value not in _evidence:
            di[value] = solution[index]
    resu = -calculate_gbf(_model, di, _evidence)
    # print(resu)
    return resu

def _categorical_cost_function_more_targets(solution):
    global _model, _evidence, _states, _possible_states, _segmented_solutions, _more_targets

    table_limit = 100000000

    di = {}
    for index, value in enumerate(_states):
        if solution[index] != 'None' and value not in _evidence:
            di[value] = solution[index]

    val = 1
    for n in di.keys():
        val *= _possible_states[n]
    
    if val < table_limit:
        resu = -calculate_gbf(_model, di, _evidence)
        return resu
    
    if _more_targets == 2:
        max_val = 1
        new_di = {}
        while max_val < table_limit:
            max_elem = random.choice([x for x in di.keys() if x not in new_di.keys()])
            if max_val*_possible_states[max_elem] > table_limit:
                break
            new_di[max_elem] = di[max_elem]
            max_val *= _possible_states[max_elem]
        
        _segmented_solutions[tuple(di)] = new_di

        resu = -calculate_gbf(_model, new_di, _evidence)
    else:
        gbfs = {}
        for node in di:
            gbfs[node] = calculate_gbf(_model, {node: di[node]}, _evidence)

        max_val = 1
        new_di = {}
        while max_val < table_limit:
            max_elem = max(gbfs, key=gbfs.get)
            if max_val*_possible_states[max_elem] > table_limit:
                break
            gbfs.pop(max_elem)
            new_di[max_elem] = di[max_elem]
            max_val *= _possible_states[max_elem]
        
        _segmented_solutions[tuple(di)] = new_di

        resu = -calculate_gbf(_model, new_di, _evidence)

    return resu





def _preprocess_mre_eda(model,states):
    all_states = list(model.states.keys())

    means = []
    for s in states:
        l = model.get_cpds(s).values
        medias = [np.mean(arr) for arr in l]
        means.append(medias)

    for a in means:
        a.append(0)

    index = [all_states.index(e) for e in all_states]
    v = list(model.states.values())
    v = [v[i] for i in index]

    a = []
    for li in v:
        a.append(np.array(li))
    hola = [x.tolist() for x in a]
    for a in hola:
        a.append('None')

    return (len(states),hola,means)


def ebna_mre(model,evidence,target=None,size_gen=100,max_iter=50,dead_iter=20,alpha=0.5,verbose=True,best_init=False,more_targets=0):
    """
    Computes mre with EBNA algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    size_gen : int, default: 100
        size of each generation
    max_iter : int, default: 50
        maximun number of iterations
    dead_iter : int, default: 20
        number of iterations without improvement
    alpha : float, default: 0.5
        percentage of population selected to update the probabilistic model.
    verbose : bool, default: True
        true to show information of the process
    best_init : bool, default: False
        if true the first generation is initialized in a efficient way to improve performance and speed
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    - elapsed time
    """
    global _states, _model, _evidence, _inference, _more_targets, _possible_states,_segmented_solutions
    _inference = VariableElimination(model)
    states = list(model.states.keys())
    states = [x for x in states if x not in evidence.keys()]
    if target!=None:
        states = target

    _states = states
    _model = model
    _evidence = evidence

    n_variables,values,freq = _preprocess_mre_eda(model,states)

    init_data = None
    if best_init:
        init_data = []
        for n in range(0,size_gen):
            c = ['None']*n_variables

            # Robust num_to_select calculation
            current_n_variables = n_variables
            min_select = 2
            max_select_from_rule = current_n_variables // 2
            min_select = min(min_select, current_n_variables)
            max_select_actual = max(min_select, max_select_from_rule)
            max_select_actual = min(max_select_actual, current_n_variables)
            
            # Ensure range is valid for randrange
            if min_select > max_select_actual:
                num_to_select = min_select # Fallback to minimum
            else:
                num_to_select = random.randrange(min_select, max_select_actual + 1)

            ind = random.sample(range(len(values)), num_to_select)
            
            for i in ind:
                c[i] = values[i][random.randrange(0,len(values[i])-1)]
            init_data.append(np.array(c))
        init_data = np.array(init_data)

    ebna = EBNA(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter, n_variables=n_variables, alpha=alpha,
            possible_values=values, frequency=freq,init_data=init_data)
    
    warnings.filterwarnings("ignore")
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        ebna_result = ebna.minimize(_categorical_cost_function_more_targets, verbose)
    else:
        ebna_result = ebna.minimize(_categorical_cost_function, verbose)

    ebna_result.best_ind
    result = {}
    for index, elem in enumerate(states):
        if ebna_result.best_ind[index] != 'None' and elem not in evidence:
            result[elem] = ebna_result.best_ind[index]

    if tuple(result) in _segmented_solutions:
        result = _segmented_solutions[tuple(result)]

    if verbose:
        print(f'Resultado: {result}')
        print(f'GBF score: {100 - ebna_result.best_cost:.4f}')
        print(f'Tiempo CPU: {ebna_result.cpu_time}')

    return (result,-ebna_result.best_cost,ebna_result.cpu_time)


def UMDAcat_mre(model,evidence,target=None,size_gen=20,max_iter=50,dead_iter=10,alpha=0.5,verbose=True,best_init=False,more_targets=0):
    """
    Computes mre with UMDAcat algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    size_gen : int, default: 100
        size of each generation
    max_iter : int, default: 50
        maximun number of iterations
    dead_iter : int, default: 20
        number of iterations without improvement
    alpha : float, default: 0.5
        percentage of population selected to update the probabilistic model.
    verbose : bool, default: True
        true to show information of the process
    best_init : bool, default: False
        if true the first generation is initialized in a efficient way to improve performance and speed
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    - elapsed time
    """
    global _states, _model, _evidence, _inference, _possible_states, _segmented_solutions, _more_targets
    _inference = VariableElimination(model)
    states = list(model.states.keys())
    states = [x for x in states if x not in evidence.keys()]
    if target!=None:
        states = target

    _states = states
    _model = model
    _evidence = evidence

    n_variables,values,freq = _preprocess_mre_eda(model,states)

    init_data = None
    if best_init:
        init_data = []
        for n in range(0,size_gen):
            c = ['None']*n_variables

            # Robust num_to_select calculation
            current_n_variables = n_variables
            min_select = 2
            max_select_from_rule = current_n_variables // 2
            min_select = min(min_select, current_n_variables)
            max_select_actual = max(min_select, max_select_from_rule)
            max_select_actual = min(max_select_actual, current_n_variables)
            
            # Ensure range is valid for randrange
            if min_select > max_select_actual:
                num_to_select = min_select # Fallback to minimum
            else:
                num_to_select = random.randrange(min_select, max_select_actual + 1)
            
            ind = random.sample(range(len(values)), num_to_select)
            
            for i in ind:
                c[i] = values[i][random.randrange(0,len(values[i])-1)]
            init_data.append(np.array(c))
        init_data = np.array(init_data)

    ebna = UMDAcat(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter, n_variables=n_variables, alpha=alpha,
            possible_values=values, frequency=freq,init_data=init_data)
    
    warnings.filterwarnings("ignore")
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        ebna_result = ebna.minimize(_categorical_cost_function_more_targets, verbose)
    else:
        ebna_result = ebna.minimize(_categorical_cost_function, verbose)

    
    result = {}
    for index, elem in enumerate(states):
        if ebna_result.best_ind[index] != 'None' and elem not in evidence:
            result[elem] = ebna_result.best_ind[index]

    if tuple(result) in _segmented_solutions:
        result = _segmented_solutions[tuple(result)]

    if verbose:
        print(f'Resultado: {result}')
        print(f'GBF score: {100 - ebna_result.best_cost:.4f}')
        print(f'Tiempo CPU: {ebna_result.cpu_time}')

    return (result,-ebna_result.best_cost,ebna_result.cpu_time)

def UMDAcat_mre2(model,evidence,target=None,size_gen=20,max_iter=50,dead_iter=10,alpha=0.5,verbose=True,best_init=False,more_targets=0):
    """
    Computes mre with UMDAcat algorithm, improved by adding all one state results

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    size_gen : int, default: 100
        size of each generation
    max_iter : int, default: 50
        maximun number of iterations
    dead_iter : int, default: 20
        number of iterations without improvement
    alpha : float, default: 0.5
        percentage of population selected to update the probabilistic model.
    verbose : bool, default: True
        true to show information of the process
    best_init : bool, default: False
        if true the first generation is initialized in a efficient way to improve performance and speed
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved\n 
    Returns
    ----------
    - dict with the solution
    - gbf value
    - elapsed time
    """
    global _states, _model, _evidence, _inference, _possible_states, _segmented_solutions, _more_targets
    _inference = VariableElimination(model)
    states = list(model.states.keys())
    states = [x for x in states if x not in evidence.keys()]
    if target!=None:
        states = target

    _states = states
    _model = model
    _evidence = evidence

    n_variables,values,freq = _preprocess_mre_eda(model,states)

    init_data = None
    if best_init:
        init_data = []
        for n in range(0,size_gen):
            c = ['None']*n_variables

            # Robust num_to_select calculation
            current_n_variables = n_variables
            min_select = 2
            max_select_from_rule = current_n_variables // 2
            min_select = min(min_select, current_n_variables)
            max_select_actual = max(min_select, max_select_from_rule)
            max_select_actual = min(max_select_actual, current_n_variables)
            
            # Ensure range is valid for randrange
            if min_select > max_select_actual:
                num_to_select = min_select # Fallback to minimum
            else:
                num_to_select = random.randrange(min_select, max_select_actual + 1)
            
            ind = random.sample(range(len(values)), num_to_select)
            
            for i in ind:
                c[i] = values[i][random.randrange(0,len(values[i])-1)]
            init_data.append(np.array(c))
        init_data = np.array(init_data)

    ebna = UMDAcat(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter, n_variables=n_variables, alpha=alpha,
            possible_values=values, frequency=freq,init_data=init_data)
    
    warnings.filterwarnings("ignore")
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        ebna_result = ebna.minimize(_categorical_cost_function_more_targets, verbose)
    else:
        ebna_result = ebna.minimize(_categorical_cost_function, verbose)

    
    result = {}
    for index, elem in enumerate(states):
        if ebna_result.best_ind[index] != 'None' and elem not in evidence:
            result[elem] = ebna_result.best_ind[index]

    if tuple(result) in _segmented_solutions:
        result = _segmented_solutions[tuple(result)]

    if verbose:
        print(f'Resultado: {result}')
        print(f'GBF score: {100 - ebna_result.best_cost:.4f}')
        print(f'Tiempo CPU: {ebna_result.cpu_time}')

    openList = initial_list(model,target)
    tagGBF = [calculate_gbf(model,x,evidence) for x in openList]
    best = (openList[tagGBF.index(max(tagGBF))],max(tagGBF))
    if -ebna_result.best_cost > best[1]:
        return (result,-ebna_result.best_cost,ebna_result.cpu_time)
    
    return (best[0],best[1],ebna_result.cpu_time)

class Codification():
    def __init__(self,model,target_states):
        self.model = model
        self.target_states = target_states
        values = []
        values_cod = []
        states = self.preprocess_mre_eda(model,target_states)
        for i,x in enumerate(target_states):
            v = states[i]
            values.append(v)
            v_cod = [x for x in range(0,len(v))]
            values_cod.append(v_cod)
        self.values = values
        self.values_cod = values_cod

    def preprocess_mre_eda(self,model,states):
        all_states = list(model.states.keys())

        means = []
        for s in states:
            l = model.get_cpds(s).values
            medias = [np.mean(arr) for arr in l]
            means.append(medias)

        for a in means:
            a.append(0)

        index = [all_states.index(e) for e in states if e in all_states]
        v = list(model.states.values())
        v = [v[i] for i in index]

        a = []
        for li in v:
            a.append(np.array(li))
        hola = [x.tolist() for x in a]
        for a in hola:
            a.append('None')

        return hola

    def encode(self, target_values):
        result = []
        for index, elem in enumerate(target_values):
            v = self.values[index]
            v_cod = self.values_cod[index]
            i = v.index(elem)
            result.append(v_cod[i])
        return result

    def decode(self,target_values):
        result = []
        for index, elem in enumerate(target_values):
            v = self.values[index]
            v_cod = self.values_cod[index]
            i = v_cod.index(round(elem))
            result.append(v[i])
        return result

class MREProblem(Problem):
    def __init__(self,model,target_states,evidence,codification,more_targets=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.target_states = target_states
        self.evidence =  evidence
        self.codificacion = codification
        self.possible_states = {x:len(y) for x,y in model.states.items()}
        self.segmented_solutions = []
        self.more_targets = more_targets

    def _evaluate(self, x, out, *args, **kwargs):
        res = []
        table_limit = 100000000
        for elem in x:
            elem = self.codificacion.decode(elem)
            di = {}
            for index, value in enumerate(self.target_states):
                if elem[index] != 'None' and value not in self.evidence:
                    di[value] = elem[index]

            if self.more_targets == 0:
                resu = -self._calculate_gbf(self.model, di, self.evidence)
                res.append(resu)
                continue

            val = 1
            for n in di.keys():
                val *= self.possible_states[n]
            
            if val < table_limit:
                resu = -self._calculate_gbf(self.model, di, self.evidence)
                res.append(resu)
                continue
            
            if self.more_targets == 2:
                max_val = 1
                new_di = {}
                while max_val < table_limit:
                    max_elem = random.choice([x for x in di.keys() if x not in new_di.keys()])
                    if max_val*self.possible_states[max_elem] > table_limit:
                        break
                    new_di[max_elem] = di[max_elem]
                    max_val *= self.possible_states[max_elem]
                
                self.segmented_solutions[tuple(di)] = new_di

                resu = -self._calculate_gbf(self.model, di, self.evidence)
            else:
                gbfs = {}
                for node in di:
                    gbfs[node] = self._calculate_gbf(self.model, {node: di[node]}, self.evidence)

                max_val = 1
                new_di = {}
                while max_val < table_limit:
                    max_elem = max(gbfs, key=gbfs.get)
                    if max_val*self.possible_states[max_elem] > table_limit:
                        break
                    gbfs.pop(max_elem)
                    new_di[max_elem] = di[max_elem]
                    max_val *= self.possible_states[max_elem]
                
                self.segmented_solutions[tuple(di)] = new_di

                resu = -self._calculate_gbf(self.model, new_di, self.evidence)

            res.append(resu)

        out['F'] = np.array(res)
    

    def _calculate_gbf(self,model, explanation, evidence):
        _inference = VariableElimination(model)

        phi_query1 = _inference.query(explanation,evidence=evidence)
        phi_query2 = _inference.query(explanation)
        p1 = phi_query1.get_value(**explanation)
        p2 = phi_query2.get_value(**explanation)

        if (p2*(1-p1)) == 0:
            return 0

        gbf = (p1*(1-p2))/(p2*(1-p1))
        return gbf

def nsga2_mre(model,evidence,target,pop_size=50,n_gen=50,best_init=False,period=20,more_targets=0):
    """
    Computes mre with NSGA2 algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    pop_size : int, default: 50
        size of each population
    n_gen : int, default: 50
        maximun number of generations
    best_init : bool, default: False
        if true the first generation is initialized in a efficient way to improve performance and speed
    period : int, default: 20
        number of generations without improvement
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    cod = Codification(model,target)

    min_values = [0]*len(target)
    max_values = []
    for e in target:
        max_values.append(len(model.states[e]))


    problem = MREProblem(model,target,evidence,cod,n_var=len(target),xl=min_values,xu=max_values,type_var=int,more_targets=more_targets)
    
    init_data = None
    if best_init:
        init_data = []
        for n in range(0,pop_size):
            c = ['None']*len(target)
            current_n_variables = len(target)

            # Robust num_to_select calculation
            min_select = 2
            max_select_from_rule = current_n_variables // 2
            min_select = min(min_select, current_n_variables)
            max_select_actual = max(min_select, max_select_from_rule)
            max_select_actual = min(max_select_actual, current_n_variables)
            
            # Ensure range is valid for randrange
            if min_select > max_select_actual:
                num_to_select = min_select # Fallback to minimum
            else:
                num_to_select = random.randrange(min_select, max_select_actual + 1)
            
            ind = random.sample(range(current_n_variables), num_to_select)

            for i in ind:
                c[i] = model.states[target[i]][random.randrange(0,len(model.states[target[i]]))]
            init_data.append(np.array(cod.encode(c)))
        init_data = np.array(init_data)

    if best_init:
        algo = NSGA2(pop_size=pop_size,sampling=init_data)
    else:
        algo = NSGA2(pop_size=pop_size)

    # stop = ('n_gen',n_gen)
    stop = DefaultSingleObjectiveTermination(
        ftol=1e-6,
        period=period,
        n_max_gen=n_gen,
    )

    result = minimize(problem=problem,algorithm=algo,termination=stop)
    print(result.F)
    gbf = min(result.F[0])
    sol_cod = result.X[0]
    sol_dec = cod.decode(sol_cod)
    sol = {c:v for c,v in zip(target,sol_dec) if v != 'None'}
    return sol,-gbf


class TabuSearch:
    """
    Conducts tabu search
    """
    __metaclass__ = ABCMeta

    cur_steps = None

    tabu_size = None
    tabu_list = None

    initial_state = None
    current = None
    best = None

    max_steps = None
    max_score = None

    def __init__(self, initial_state, tabu_size, max_steps, max_score=None):
        """

        :param initial_state: initial state, should implement __eq__ or __cmp__
        :param tabu_size: number of states to keep in tabu list
        :param max_steps: maximum number of steps to run algorithm for
        :param max_score: score to stop algorithm once reached
        """
        self.initial_state = initial_state

        if isinstance(tabu_size, int) and tabu_size > 0:
            self.tabu_size = tabu_size
        else:
            raise TypeError('Tabu size must be a positive integer')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise TypeError('Maximum steps must be a positive integer')

        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')

    def __str__(self):
        return ('TABU SEARCH: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST SCORE: %f \n' +
                'BEST MEMBER: %s \n\n') % \
               (self.cur_steps, self._score(self.best), str(self.best))

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm

        :return: None
        """
        self.cur_steps = 0
        self.tabu_list = deque(maxlen=self.tabu_size)
        self.current = self.initial_state
        self.best = self.initial_state

    @abstractmethod
    def _score(self, state):
        """
        Returns objective function value of a state

        :param state: a state
        :return: objective function value of state
        """
        pass

    @abstractmethod
    def _neighborhood(self):
        """
        Returns list of all members of neighborhood of current state, given self.current

        :return: list of members of neighborhood
        """
        pass

    def _best(self, neighborhood):
        """
        Finds the best member of a neighborhood

        :param neighborhood: a neighborhood
        :return: best member of neighborhood
        """
        return neighborhood[argmax([self._score(x) for x in neighborhood])]

    def run(self, verbose=True):
        """
        Conducts tabu search

        :param verbose: indicates whether or not to print progress regularly
        :return: best state and objective function value of best state
        """
        self._clear()
        for i in range(self.max_steps):
            self.cur_steps += 1

            if ((i + 1) % 100 == 0) and verbose:
                print(self)

            neighborhood = self._neighborhood()
            neighborhood_best = self._best(neighborhood)

            while True:
                if all([x in self.tabu_list for x in neighborhood]):
                    print("TERMINATING - NO SUITABLE NEIGHBORS")
                    return self.best, self._score(self.best)
                if neighborhood_best in self.tabu_list:
                    if self._score(neighborhood_best) > self._score(self.best):
                        self.tabu_list.append(neighborhood_best)
                        self.best = deepcopy(neighborhood_best)
                        break
                    else:
                        neighborhood.remove(neighborhood_best)
                        neighborhood_best = self._best(neighborhood)
                else:
                    self.tabu_list.append(neighborhood_best)
                    self.current = neighborhood_best
                    if self._score(self.current) > self._score(self.best):
                        self.best = deepcopy(self.current)
                    break

            if self.max_score is not None and self._score(self.best) > self.max_score:
                print("TERMINATING - REACHED MAXIMUM SCORE")
                return self.best, self._score(self.best)
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.best, self._score(self.best)


class Tabu_mre(TabuSearch):
    def __init__(self, model,evidence,target,tabu_size, max_steps, max_score=None):
        _,values,_ = Tabu_mre._preprocess_mre_eda(model,target)
        self.values = values
        c = ['None']*len(target)

        current_n_variables = len(target)
        # Robust num_to_select calculation for Tabu_mre __init__
        min_select = 2
        max_select_from_rule = current_n_variables // 2
        min_select = min(min_select, current_n_variables)
        max_select_actual = max(min_select, max_select_from_rule)
        max_select_actual = min(max_select_actual, current_n_variables)
        
        # Ensure range is valid for randrange
        if min_select > max_select_actual:
            num_to_select = min_select # Fallback to minimum
        else:
            num_to_select = random.randrange(min_select, max_select_actual + 1)
        
        ind = random.sample(range(len(values)), num_to_select)

        for i in ind:
            c[i] = values[i][random.randrange(0,len(values[i])-1)]
        initial_state = c
        super().__init__(initial_state, tabu_size, max_steps, max_score)
        self.model = model
        self.evidence = evidence
        self.target = target
        self.inference = VariableElimination(model)

    def _score(self, state):
        global _more_targets, _segmented_solutions, _possible_states
        di = {}
        for index, value in enumerate(self.target):
            if state[index] != 'None' and value not in self.evidence:
                di[value] = state[index]

        if _more_targets == 0:
            resu = self.calculate_gbf(self.model, di, self.evidence)
            return resu
        
        table_limit = 100000000

        val = 1
        for n in di.keys():
            val *= _possible_states[n]
        
        if val < table_limit:
            resu = self.calculate_gbf(self.model, di, self.evidence)
            return resu

        if _more_targets == 2:
            max_val = 1
            new_di = {}
            while max_val < table_limit:
                max_elem = random.choice([x for x in di.keys() if x not in new_di.keys()])
                if max_val*_possible_states[max_elem] > table_limit:
                    break
                new_di[max_elem] = di[max_elem]
                max_val *= _possible_states[max_elem]
            
            _segmented_solutions[tuple(di)] = new_di

            resu = self.calculate_gbf(self.model, di, self.evidence)
        else:
            gbfs = {}
            for node in di:
                gbfs[node] = self.calculate_gbf(self.model, {node: di[node]}, self.evidence)

            max_val = 1
            new_di = {}
            while max_val < table_limit:
                max_elem = max(gbfs, key=gbfs.get)
                if max_val*_possible_states[max_elem] > table_limit:
                    break
                gbfs.pop(max_elem)
                new_di[max_elem] = di[max_elem]
                max_val *= _possible_states[max_elem]
            
            _segmented_solutions[tuple(di)] = new_di

            resu = self.calculate_gbf(self.model, new_di, self.evidence)

        return resu
    
    def _neighborhood(self):
        poss_states = []
        for index,state in enumerate(self.target):
            for value in self.values[index]:
                if value != self.current[index]:
                    new_states = self.current.copy()
                    new_states[index] = value
                    poss_states.append(new_states)
        return poss_states
    
    def _preprocess_mre_eda(model,states):
        all_states = list(model.states.keys())

        means = []
        for s in states:
            l = model.get_cpds(s).values
            medias = [np.mean(arr) for arr in l]
            means.append(medias)

        for a in means:
            a.append(0)

        index = [all_states.index(e) for e in all_states]
        v = list(model.states.values())
        v = [v[i] for i in index]

        a = []
        for li in v:
            a.append(np.array(li))
        hola = [x.tolist() for x in a]
        for a in hola:
            a.append('None')

        return (len(states),hola,means)

    def calculate_gbf(self, model, explanation, evidence):
        phi_query1 = self.inference.query(explanation,evidence=evidence)
        phi_query2 = self.inference.query(explanation)
        p1 = phi_query1.get_value(**explanation)
        p2 = phi_query2.get_value(**explanation)

        if (p2*(1-p1)) == 0:
            return 0

        gbf = (p1*(1-p2))/(p2*(1-p1))
        return gbf


def tabu_mre(model,evidence,target,tabu_size, max_steps, max_score=None,more_targets=0):
    """
    Computes mre with Tabu algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    tabu_size : int
        number of states to keep in tabu list
    max_steps : int
        maximun number of steps
    max_score : float, default: None
        maximun score, if achieved the algorithm will terminate
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    global _more_targets,_possible_states,_segmented_solutions
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
    tabu = Tabu_mre(model,evidence,target,tabu_size,max_steps,max_score)
    sol_val, gbf = tabu.run()
    # sol = {x:y for x,y in zip(target,sol_val)}
    di = {}
    for index, value in enumerate(target):
        if sol_val[index] != 'None' and value not in evidence:
            di[value] = sol_val[index]

    if tuple(di) in _segmented_solutions:
        di = _segmented_solutions[tuple(di)]
    return di, gbf




def generate_initial_sample(random, args):
    model = args.get('model')
    target = args.get('target')
    cod = args.get('cod')
    c = ['None']*len(target)

    current_n_variables = len(target)
    # Robust num_to_select calculation for generate_initial_sample
    min_select = 2
    max_select_from_rule = current_n_variables // 2
    min_select = min(min_select, current_n_variables)
    max_select_actual = max(min_select, max_select_from_rule)
    max_select_actual = min(max_select_actual, current_n_variables)
    
    # Ensure range is valid for randrange
    if min_select > max_select_actual:
        num_to_select = min_select # Fallback to minimum
    else:
        num_to_select = random.randrange(min_select, max_select_actual + 1)
    
    ind = random.sample(range(current_n_variables), num_to_select)

    for i in ind:
        c[i] = model.states[target[i]][random.randrange(0,len(model.states[target[i]]))]
    c = cod.encode(c)
    return c

def calculate_gbf(model, explanation, evidence):
    _inference = VariableElimination(model)

    phi_query1 = _inference.query(explanation,evidence=evidence)
    phi_query2 = _inference.query(explanation)
    p1 = phi_query1.get_value(**explanation)
    p2 = phi_query2.get_value(**explanation)

    if (p2*(1-p1)) == 0:
        return 0

    gbf = (p1*(1-p2))/(p2*(1-p1))
    return gbf

def evaluate_sample(candidates, args):
    model = args.get('model')
    target = args.get('target')
    cod = args.get('cod')
    evidence = args.get('evidence')
    res = []
    for elem in candidates:
        elem = cod.decode(elem)
        di = {}
        for index, value in enumerate(target):
            if elem[index] != 'None' and value not in evidence:
                di[value] = elem[index]
        resu = calculate_gbf(model, di, evidence)
        res.append(resu)
    return res

def evaluate_sample2(candidates, args):
    global _more_targets,_segmented_solutions,_possible_states
    model = args.get('model')
    target = args.get('target')
    cod = args.get('cod')
    evidence = args.get('evidence')
    res = []
    table_limit = 100000000
    for elem in candidates:
        elem = cod.decode(elem)
        di = {}
        for index, value in enumerate(target):
            if elem[index] != 'None' and value not in evidence:
                di[value] = elem[index]

        val = 1
        for n in di.keys():
            val *= _possible_states[n]
        
        if val < table_limit:
            resu = calculate_gbf(model, di, evidence)
            res.append(resu)
            continue
        
        if _more_targets == 2:
            max_val = 1
            new_di = {}
            while max_val < table_limit:
                max_elem = random.choice([x for x in di.keys() if x not in new_di.keys()])
                if max_val*_possible_states[max_elem] > table_limit:
                    break
                new_di[max_elem] = di[max_elem]
                max_val *= _possible_states[max_elem]
            
            _segmented_solutions[tuple(di)] = new_di

            resu = calculate_gbf(model, new_di, evidence)
        else:
            gbfs = {}
            for node in di:
                gbfs[node] = calculate_gbf(model, {node: di[node]}, evidence)

            max_val = 1
            new_di = {}
            while max_val < table_limit:
                max_elem = max(gbfs, key=gbfs.get)
                if max_val*_possible_states[max_elem] > table_limit:
                    break
                gbfs.pop(max_elem)
                new_di[max_elem] = di[max_elem]
                max_val *= _possible_states[max_elem]
            
            _segmented_solutions[tuple(di)] = new_di

            resu = calculate_gbf(model, new_di, evidence)
        
        res.append(resu)

    return res

def bound_sample(candidate, args):
    upper_bound = args.get('upper_bound')
    for i,e in enumerate(candidate):
        candidate[i] = max(min(e,upper_bound[i]),0)
    return candidate


def ga_mre(model,evidence,target,pop_size=50,n_gen=50,more_targets=0):
    """
    Computes mre with Genetic algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    pop_size : int, default: 50
        size of each population
    n_gen : int, default: 50
        maximun number of generations
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    global _more_targets,_possible_states,_segmented_solutions
    cod = Codification(model,target)
    global _inference
    _inference = VariableElimination(model)

    max_values = []
    for e in target:
        max_values.append(len(model.states[e]))

    rand = random.Random()
    ea = inspyred.ec.GA(rand)
    ea.terminator = [terminators.evaluation_termination,inspyred.ec.terminators.no_improvement_termination]
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        final_pop = ea.evolve(generator=generate_initial_sample,
                            evaluator=evaluate_sample2,
                            pop_size=pop_size,
                            maximize=True,
                            bounder=bound_sample,
                            max_evaluations=n_gen,
                            mutation_rate=0.25,
                            model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values,max_generations=100)
    else:
        final_pop = ea.evolve(generator=generate_initial_sample,
                            evaluator=evaluate_sample,
                            pop_size=pop_size,
                            maximize=True,
                            bounder=bound_sample,
                            max_evaluations=n_gen,
                            mutation_rate=0.25,
                            model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values,max_generations=100)
    # Sort and print the best individual, who will be at index 0.
    final_pop.sort(reverse=True)
    sol = cod.decode(final_pop[0].candidate)
    di = {}
    for index, value in enumerate(target):
        if sol[index] != 'None' and value not in evidence:
            di[value] = sol[index]
    if tuple(di) in _segmented_solutions:
        di = _segmented_solutions[tuple(di)]
    gbf = final_pop[0].fitness
    return di,gbf

def es_mre(model,evidence,target,pop_size=50,n_gen=50,more_targets=0):
    """
    Computes mre with Evolution Strategy algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    pop_size : int, default: 50
        size of each population
    n_gen : int, default: 50
        maximun number of generations
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    global _more_targets,_possible_states,_segmented_solutions
    cod = Codification(model,target)
    global _inference
    _inference = VariableElimination(model)

    max_values = []
    for e in target:
        max_values.append(len(model.states[e]))

    rand = random.Random()
    ea = inspyred.ec.ES(rand)
    ea.terminator = [terminators.evaluation_termination,inspyred.ec.terminators.no_improvement_termination]
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        final_pop = ea.evolve(generator=generate_initial_sample,
                            evaluator=evaluate_sample2,
                            pop_size=pop_size,
                            maximize=True,
                            bounder=bound_sample,
                            max_evaluations=n_gen,
                            model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values)
    else:
        final_pop = ea.evolve(generator=generate_initial_sample,
                        evaluator=evaluate_sample,
                        pop_size=pop_size,
                        maximize=True,
                        bounder=bound_sample,
                        max_evaluations=n_gen,
                        model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values)
    # Sort and print the best individual, who will be at index 0.
    final_pop.sort(reverse=True)
    sol = cod.decode(final_pop[0].candidate[0:len(final_pop[0].candidate)//2])
    di = {}
    for index, value in enumerate(target):
        if sol[index] != 'None' and value not in evidence:
            di[value] = sol[index]
    if tuple(di) in _segmented_solutions:
        di = _segmented_solutions[tuple(di)]
    gbf = final_pop[0].fitness
    return di,gbf

def dea_mre(model,evidence,target,pop_size=50,n_gen=50,more_targets=0):
    """
    Computes mre with Differential Evolution algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    pop_size : int, default: 50
        size of each population
    n_gen : int, default: 50
        maximun number of generations
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    global _more_targets,_possible_states,_segmented_solutions
    cod = Codification(model,target)
    global _inference
    _inference = VariableElimination(model)

    max_values = []
    for e in target:
        max_values.append(len(model.states[e]))

    rand = random.Random()
    ea = inspyred.ec.DEA(rand)
    ea.terminator = terminators.evaluation_termination
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        final_pop = ea.evolve(generator=generate_initial_sample,
                            evaluator=evaluate_sample2,
                            pop_size=pop_size,
                            maximize=True,
                            bounder=bound_sample,
                            max_evaluations=n_gen,
                            model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values)
    else:
        final_pop = ea.evolve(generator=generate_initial_sample,
                    evaluator=evaluate_sample,
                    pop_size=pop_size,
                    maximize=True,
                    bounder=bound_sample,
                    max_evaluations=n_gen,
                    model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values)
    # Sort and print the best individual, who will be at index 0.
    final_pop.sort(reverse=True)
    sol = cod.decode(final_pop[0].candidate)
    di = {}
    for index, value in enumerate(target):
        if sol[index] != 'None' and value not in evidence:
            di[value] = sol[index]
    if tuple(di) in _segmented_solutions:
        di = _segmented_solutions[tuple(di)]
    gbf = final_pop[0].fitness
    return di,gbf

def pso_mre(model,evidence,target,pop_size=50,n_gen=50,more_targets=0):
    """
    Computes mre with Particle Swarm Optimization algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target : list
        list of target nodes
    pop_size : int, default: 50
        size of each population
    n_gen : int, default: 50
        maximun number of generations
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved \n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    global _more_targets,_possible_states,_segmented_solutions
    cod = Codification(model,target)
    global _inference
    _inference = VariableElimination(model)

    max_values = []
    for e in target:
        max_values.append(len(model.states[e]))

    rand = random.Random()
    ea = inspyred.ec.DEA(rand)
    ea.terminator = terminators.evaluation_termination
    if more_targets != 0:
        _more_targets = more_targets
        _possible_states = {x:len(y) for x,y in model.states.items()}
        _segmented_solutions = {}
        final_pop = ea.evolve(generator=generate_initial_sample,
                            evaluator=evaluate_sample2,
                            pop_size=pop_size,
                            maximize=True,
                            bounder=bound_sample,
                            max_evaluations=n_gen,
                            neighborhood_size=5,
                            model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values)
    else:
        final_pop = ea.evolve(generator=generate_initial_sample,
                            evaluator=evaluate_sample,
                            pop_size=pop_size,
                            maximize=True,
                            bounder=bound_sample,
                            max_evaluations=n_gen,
                            neighborhood_size=5,
                            model=model,target=target,evidence=evidence,cod=cod,upper_bound=max_values)
    # Sort and print the best individual, who will be at index 0.
    final_pop.sort(reverse=True)
    sol = cod.decode(final_pop[0].candidate)
    di = {}
    for index, value in enumerate(target):
        if sol[index] != 'None' and value not in evidence:
            di[value] = sol[index]
    if tuple(di) in _segmented_solutions:
        di = _segmented_solutions[tuple(di)]
    gbf = final_pop[0].fitness
    return di,gbf

def initial_list(model, target_variables):
    init_list = []
    for target in target_variables:
        states_values = model.states[target]
        for state in states_values:
            init_list.append({target:state})
    return init_list

def reduce_num_targets(model, explanation, evidence):
    global _more_targets, _segmented_solutions,_possible_states
    table_limit = 100000000

    val = 1
    for n in explanation.keys():
        val *= _possible_states[n]
    
    if val < table_limit:
        return explanation
    
    if _more_targets == 2:
        max_val = 1
        new_di = {}
        while max_val < table_limit:
            max_elem = random.choice([x for x in explanation.keys() if x not in new_di.keys()])
            if max_val*_possible_states[max_elem] > table_limit:
                break
            new_di[max_elem] = explanation[max_elem]
            max_val *= _possible_states[max_elem]
        
        _segmented_solutions[tuple(explanation)] = new_di

        return new_di
    else:
        gbfs = {}
        for node in explanation:
            gbfs[node] = calculate_gbf(model, {node: explanation[node]}, evidence)

        max_val = 1
        new_di = {}
        while max_val < table_limit:
            max_elem = max(gbfs, key=gbfs.get)
            if max_val*_possible_states[max_elem] > table_limit:
                break
            gbfs.pop(max_elem)
            new_di[max_elem] = explanation[max_elem]
            max_val *= _possible_states[max_elem]
        
        _segmented_solutions[tuple(explanation)] = new_di
        return new_di



def calculate_gbf_hbs(model, explanation, evidence):
    global _more_targets
    if _more_targets !=0:
        explanation = reduce_num_targets(model,explanation,evidence)

    _inference = VariableElimination(model)

    phi_query1 = _inference.query(explanation,evidence=evidence)
    phi_query2 = _inference.query(explanation)
    p1 = phi_query1.get_value(**explanation)
    p2 = phi_query2.get_value(**explanation)

    if (p2*(1-p1)) == 0:
        return 0

    gbf = (p1*(1-p2))/(p2*(1-p1))
    return gbf

def hierarchical_beam_search(model,evidence,target_variables,beam_size,T,N,K,more_targets=0):
    """
    Computes mre with Hierarchical Beam Search algorithm

    Parameters
    ----------
    model :
        bayesian network model for inference
    evidence : dict
        dict containing the evidence nodes and values
    target_variables : list
        list of target nodes
    beam_size : int
        size of the beam
    T : int
        supper bound of cbf, used to prune solutions. Should be a bit more than 1 like 1+e10-8
    N : int
        size of the first level beam
    K : int
        size of the second level beam
    more_targets : int, default: 0
        if 0, the number of target states is limited and could fail\n
        with another number, the number of targets is infinite, but the gbf value will be calculated with a limit of states\n
        with 1 will take the best gbf states individually, and with 2 will take random states until the limit is achieved\n
    Returns
    ----------
    - dict with the solution
    - gbf value
    """
    global _more_targets,_possible_states,_segmented_solutions
    workList = []
    workListGBF = []
    openList = initial_list(model,target_variables)
    first_list = openList.copy()
    tagGBF = [calculate_gbf_hbs(model,x,evidence) for x in openList]
    best = (openList[tagGBF.index(max(tagGBF))],max(tagGBF))
    _more_targets = more_targets
    _possible_states = {x:len(y) for x,y in model.states.items()}
    _segmented_solutions = {}
    while len(openList) > 0:
        expState = openList.pop()
        #Second level
        locList = []
        locListGBF = []
        hbeam_expan(model,target_variables,evidence,expState,locList,tagGBF,workList,T,openList,locListGBF,K,first_list)

        #update worklist con locList
        if len(workList) < N + len(locList):
            workList.extend(locList)
            workListGBF.extend(locListGBF)
        else:
            for ind, elem in enumerate(locList):
                if len(workList) < N:
                    workList.append(elem)
                    workListGBF.append(locListGBF[ind])
                else:
                    workList_sort, workListGBF_sort = zip(*sorted(zip(workList, workListGBF), key=lambda x: x[1],reverse=True))
                    workList_sort = list(workList_sort)
                    workListGBF_sort = list(workListGBF_sort)
                    if workListGBF_sort[-1] < locListGBF[ind]:
                        workList_sort[-1] = elem
                        workListGBF_sort[-1] = locListGBF[ind]
                        workList = workList_sort
                        workListGBF = workListGBF_sort


        if len(workList) > 0 and max(workListGBF) > best[1]:
            best = (workList[workListGBF.index(max(workListGBF))],max(workListGBF))


        if len(openList) == 0:
            openList = workList.copy()
            workList.clear()
            workListGBF.clear()

    if tuple(best[0]) in _segmented_solutions:
        best = (_segmented_solutions[tuple(best[0])],best[1])
    return best

def calculate_cbf(model, explanation1,explanation2, evidence):
    global _more_targets
    if _more_targets !=0:
        explanation1 = reduce_num_targets(model,explanation1,evidence)
    _inference = VariableElimination(model)

    evidence_explanation2 = evidence.copy()
    for e in explanation2.keys():
        if e not in explanation1.keys():
            evidence_explanation2[e] = explanation2[e]
    phi_query1 = _inference.query(explanation1,evidence=evidence_explanation2)
    # phi_query2 = _inference.query(explanation1,evidence=explanation2)
    phi_query2 = _inference.query(explanation1)
    p1 = phi_query1.get_value(**explanation1)
    p2 = phi_query2.get_value(**explanation1)

    if (p2*(1-p1)) == 0:
        return 0

    cbf = (p1*(1-p2))/(p2*(1-p1))
    return cbf

def calculate_r_x_e(model,expState,evidence):
    global _more_targets
    if _more_targets !=0:
        expState = reduce_num_targets(model,expState,evidence)
    _inference = VariableElimination(model)
    
    phi_query1 = _inference.query(expState,evidence=evidence)
    phi_query2 = _inference.query(expState)
    p1 = phi_query1.get_value(**expState)
    p2 = phi_query2.get_value(**expState)

    return p1/p2

def calculate_r_x_e_alter_exp(model,expState,evidence):
    global _more_targets
    if _more_targets !=0:
        expState = reduce_num_targets(model,expState,evidence)
    _inference = VariableElimination(model)
    
    phi_query1 = _inference.query(expState,evidence=evidence)
    phi_query2 = _inference.query(expState)
    p1 = phi_query1.get_value(**expState)
    p2 = phi_query2.get_value(**expState)

    return (1-p1)/(1-p2)

def hbeam_expan(model,target_variables,evidence,expState,locList,tagGBF,workList,T,openList,locListGBF,K,first_list):
    r_x_e = calculate_r_x_e(model,expState,evidence)
    expTags = [x for x in target_variables if x not in expState.keys()]
    for expTag in expTags:
        for expTagVal in model.states[expTag]:
            sucState = expState.copy()
            sucState[expTag] = expTagVal
            if sucState in workList:
                continue
            
            cbf = calculate_cbf(model,sucState,expState,evidence)
            if cbf < 1/r_x_e or cbf < T:
                continue
            gbf = calculate_gbf_hbs(model,sucState,evidence)
            if gbf < tagGBF[first_list.index({expTag:expTagVal})]:
                continue

            #update loc list con sucState
            if len(locList) < K + 1:
                locList.append(sucState)
                locListGBF.append(gbf)
            else:
                if len(locList) < K:
                    locList.append(sucState)
                    locListGBF.append(gbf)
                else:
                    locList_sort, locListGBF_sort = zip(*sorted(zip(locList,locListGBF), key=lambda x: x[1],reverse=True))
                    locList_sort = list(locList_sort)
                    locListGBF_sort = list(locListGBF_sort)
                    if locListGBF_sort[-1] < gbf:
                        locList_sort[-1] = sucState
                        locListGBF_sort[-1] = gbf
                        workList = locList_sort
                        workListGBF = locListGBF_sort