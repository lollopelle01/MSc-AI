import os
import time
import math
from SMT.SMT_utils import *
from SMT.SMT_solver import *
from SMT.z3_solver import *
from datetime import timedelta


def solve_instance(
        instance_file,
        solver,
        params,
        model_params:dict=None,
        verbose=False,
):
    match solver:
        case 'z3':
            return launch_z3(instance_file,params,model_params,verbose)
        case 'cvc5':
            return launch_cvc5(instance_file,params,model_params,verbose)
        case _:
            raise KeyError('Unsupported solver')
        

def launch_z3(instance_file,params,model_params,verbose):
    instance_data=parse_dzn(instance_file)
    model_params=check_model_params(model_params)
    
    try:
        if type(params['timeout'])==timedelta:
            params['timeout']=params['timeout'].total_seconds()
    except: pass
    solution=[]
    opt=False
    execTime = time.time()
    if model_params['z3']:
        solution,obj, optimal = solve_z3_model(params, instance_data,execTime)
        execTime = math.floor(time.time()-execTime)
        if not solution:
            execTime=math.floor(params['timeout'])
            
        try:
            if execTime >= params['timeout']:
                execTime = math.floor(params['timeout'])
                
            else:
                opt = True
            if optimal:
                opt=True
                execTime=execTime-1 if execTime>0 else execTime
            else:
                opt=False
                execTime=math.floor(params['timeout'])
            return {
                "time": execTime,
                "optimal": opt,
                "obj": obj,
                "sol": solution
            },None
        except:
            return {
                "time": execTime,
                "optimal": True,
                "obj": obj,
                "sol": solution
            },None
    else:
        model_head,model_tail=generate_model(instance_data,**model_params)
        obj='N/A'
        model=''
        aux=params.copy()
        best_model=None
        max_path=instance_data['upper_bound']
        num_couriers=instance_data['num_couriers']
        if model_params['best']:
            use_arrays=False
        else:
            use_arrays=model_params['use_arrays']
        aux['use_arrays']=use_arrays
        aux['best']=model_params['best']
        use_successors=model_params['use_successors']
        impose_lower_bound=model_params['impose_lower_bound']
        while max_path>=instance_data['lower_bound']:
            model=add_objective(
                instance_data['num_couriers'],
                max_path,
                model_head,
                model_tail,
                arrays=use_arrays,
                impose_lower_bound=impose_lower_bound)
            try:
                aux['timeout']=params['timeout']-(time.time()-execTime)
                if aux['timeout']<=0:
                    if verbose:print('no solution found in the available time')
                    break
                if verbose:print('available time:',aux['timeout'])
            except:
                print('error')
                pass
            result,sol,distances=solve_z3(model,aux,use_arrays,num_couriers)
            if result=='unsat':
                try:
                    if params['timeout']-(time.time()-execTime)>0:
                        opt=True
                        if verbose:print(f'unsat with lower bound {max_path}')
                    else:
                        if verbose:print('no solution found in the available time')
                        best_model=model
                except: opt=True
                break
            obj=max(distances)
            max_path=obj-1  
            solution=parse_solution(sol,use_arrays,'z3',best=aux['best'])
            if verbose:print(f'found {obj}, {solution}')
            best_model=model
            if max_path<instance_data['lower_bound']:
                if verbose:print(f'unsat with lower bound {max_path}')
                opt=True
        execTime = math.floor(time.time()-execTime)
        if not solution:
            execTime=math.floor(params['timeout'])
        try:
            if execTime > params['timeout']:
                execTime = math.floor(params['timeout'])
            return {
                "time": execTime,
                "optimal": opt,
                "obj": obj,
                "sol": solution
            },best_model
        except:
            return {
                "time": execTime,
                "optimal": opt,
                "obj": obj,
                "sol": solution
            },best_model
        

def generate_model(
        data,
        simmetry_method:str='>',
        use_successors=True,
        use_arrays=True,
        impose_lower_bound=False,
        redundancy=False,
        best=False,
        z3=False
):
    if best: return generate_best_model(data,simmetry_method)
    num_couriers = data['num_couriers']
    num_items = data['num_items']
    lower_bound = data['lower_bound']
    upper_bound = data['upper_bound']

    
    constants = f'''
(set-logic ALL)
(declare-fun num_couriers () Int)
(assert (= num_couriers {num_couriers}))
(declare-fun num_items () Int)
(assert (= num_items {num_items}))
(declare-fun lower_bound () Int)
(assert (= lower_bound {lower_bound}))
(declare-fun upper_bound () Int)
(assert (= upper_bound {upper_bound}))
'''.lstrip()
    
    variables=define_decision_variables(data,use_arrays)

    bin_packing=define_bin_packing(num_couriers,num_items,use_arrays)

    

    all_couriers_deliver=define_couriers_duty(num_couriers,num_items,use_arrays)

    

    simmetry=define_simmetry(num_couriers,simmetry_method,use_arrays)

    all_items_delivered=define_deliver_everything(num_couriers,num_items,use_arrays) if redundancy else ''

    channeling=define_padding(num_couriers,num_items,use_arrays)

    item_carrier=define_courier_items(num_couriers,num_items,use_arrays)

    if use_successors:
        def_successors,distances_calculation=define_successors_distances(num_couriers,num_items,impose_lower_bound,use_arrays)
    else:
        distances_calculation=define_distances(num_couriers,num_items,use_arrays)

    model_head=f'''
{constants}
{variables}
{bin_packing}
{all_couriers_deliver}
{simmetry}
{all_items_delivered}
{channeling}
{item_carrier}
{def_successors if use_successors else ''}
{distances_calculation}
'''
    model_tail='(check-sat)\n(get-model)\n'

    return model_head,model_tail

def launch_cvc5(instance_file, params, model_params, verbose):
    from multiprocessing import Process, Queue
    import time
    import math
    from datetime import timedelta

    instance_data = parse_dzn(instance_file)
    model_params = check_model_params(model_params)

    
    try:
        if isinstance(params.get('timeout'), timedelta):
            params['timeout'] = params['timeout'].total_seconds()
    except:
        pass
    
    total_timeout = params['timeout']
    solution = []
    opt = True
    generation_time = time.time()
    model_head, model_tail = generate_model(instance_data, **model_params)
    obj = 'N/A'
    model = ''
    aux = params.copy()
    best_model = None
    max_path = instance_data['upper_bound']
    num_couriers = instance_data['num_couriers']
    
    use_arrays = not model_params['best'] and model_params['use_arrays']
    aux['use_arrays'] = use_arrays
    aux['best'] = model_params['best']
    impose_lower_bound = model_params['impose_lower_bound']
    
    model = add_objective(instance_data['num_couriers'],
                        max_path,
                        model_head,
                        model_tail,
                        arrays=use_arrays,
                        impose_lower_bound=impose_lower_bound)
    elapsed_time = time.time() - generation_time
    remaining_time = total_timeout - elapsed_time
    
    solution, distances,obj =  [], [-1],-1
    while max_path >= instance_data['lower_bound'] and remaining_time > 0:
        sol_time = time.time()
        return_queue = Queue()

        p = Process(target=solve_cvc5, args=(model, use_arrays, num_couriers, aux['best'], return_queue))
        p.start()
        p.join(remaining_time)
        
        if p.is_alive():
            if verbose:print('Solver is still running... Let\'s kill it...')
            opt = False
            p.terminate()
            p.join()
        else:
            sat, solution_, distances = return_queue.get()
            if not sat:
                if solution:
                    if verbose:print(f'unsat with lower bound {max_path}')
                    elapsed_time = time.time() - sol_time
                    break
                else:
                    if verbose: print('no solution found in the available time')
                    opt = False
                    best_model = model
                    break
            obj = max(distances)
            max_path=obj-1
            if verbose:print(f'Found solution with objective {obj}: {solution_}')
            best_model = model
            solution=solution_
            if verbose:print(f'Found : {solution}')
            model = add_objective(instance_data['num_couriers'],
                              max_path,
                              model_head,
                              model_tail,
                              arrays=use_arrays,
                              impose_lower_bound=impose_lower_bound)
        elapsed_time = time.time() - sol_time
        remaining_time = remaining_time - elapsed_time
        if verbose: print(f'remaining time: {remaining_time}')
        
    execTime = math.floor(params['timeout']-(remaining_time))
    if not solution:
        execTime = math.floor(params['timeout'])
    
    if execTime > params['timeout']:
        execTime = math.floor(params['timeout'])

    return {
        "time": execTime,
        "optimal": opt,
        "obj": obj if obj!=-1 else 'N/A',
        "sol": solution
    }, best_model
