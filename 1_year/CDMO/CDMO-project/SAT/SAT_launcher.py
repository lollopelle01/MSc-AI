from datetime import timedelta
from  utils.utils import parse_dzn
import SAT.SAT_solver as SAT_solver
from SAT.SAT_utils import *
import time
import math
import numpy as np

def solve_instance(
        instance_file,
        model,                                            
        search_strategy,                                           
        params,
        model_params,
        verbose_search=False, 
        verbose_solver=False
):
    instance_data=parse_dzn(instance_file)
    
    # Check if the distances are symmetrycal
    m = np.array(instance_data['distances'])
    distance_symmetry = np.all(m==m.T)
    
    try:
        if type(params['timeout'])==timedelta:
            params['timeout']=params['timeout'].total_seconds()
    except: pass
    
    execTime = time.time()
    
    obj, solution = solve_strategy(instance_data, 
                                   model, 
                                   search_strategy, 
                                   params, 
                                   execTime, 
                                   **(model_params), 
                                   distance_symmetry=distance_symmetry,
                                   verbose_search=verbose_search, 
                                   verbose_solver=verbose_solver
                                )
                    
    execTime = math.floor(time.time()-execTime)
    
    if not solution:
        execTime=math.floor(params['timeout'])
        solution=[]
    try:
        if execTime > params['timeout']:
            execTime = math.floor(params['timeout'])
        return {
            "time": execTime,
            "optimal": (execTime < params['timeout']),
            "obj": obj,
            "sol": solution
        }
    except:
        return {
            "time": execTime,
            "optimal": True,
            "obj": obj,
            "sol": solution
        }