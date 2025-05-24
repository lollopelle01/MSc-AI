from MIP.MIP_solver import *
from utils  import *
from datetime import timedelta
import math

def solve_instance(
        instance_file,
        solver,
        model_name,
        params,
        model_params:dict =None,
        verbose =False,
):
    instance_data =parse_dzn(instance_file)
    obj ='N/A'
    try:
        if type(params['timeout']) == timedelta:
            params['timeout'] = params['timeout'].total_seconds()
    except: pass    
    
    solution = []
    best_model = None
    opt = False

    solution, obj, opt, execTime = solve(solver, model_name, params, instance_data, verbose)
    try:
        return {
            "time": execTime,
            "optimal": opt,
            "obj": obj,
            "sol": solution
        },best_model
    except:
        return {
            "time": execTime,
            "optimal": True,
            "obj": obj,
            "sol": solution
        },best_model