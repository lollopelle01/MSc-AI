from utils import *
import time as t
from z3 import  *

def evaluate(model, model_variables, num_couriers, num_items):
    loads, distances_traveled, stops, item_resp, successors = model_variables
    loads = [model.evaluate(loads[c]) for c in range(num_couriers)]
    item_resp = [int(str(model.evaluate(item_resp[i]))) +1 for i in range(num_items)]
    distances_traveled = [int(str(model.evaluate(distances_traveled[c]))) for c in range(num_couriers)]
    successors = [int(str(model.evaluate(successors[i]))) +1 for i in range(num_items)]
    stops = [[int(str(model.evaluate(stops[c][i]))) +1 for i in range(num_items-num_couriers+2)] for c in range(num_couriers)]
    # print('loads:',loads,'\ndists:',distances_traveled,'\nresp:',item_resp,'\nsucc:',successors,'\nstops:')
    # for row in stops:
    #     print('     ',row)
    
    return stops, distances_traveled

def solve_z3_model(params:dict, data, start_time):
    import time
    # import math

    num_couriers = data['num_couriers']
    num_items = data['num_items']


    optimizer = Optimize()
    # aux_time=time.time()
    # objective, model_variables = set_model(data, optimizer)
    objective, model_variables = set_default_model(data, optimizer)
    pars=params.copy()
    # print(time.time()-start_time)
    # print(int((pars['timeout']-(time.time()-start_time))*1000))
    if 'timeout' in params:
        pars['timeout']=int((pars['timeout']-(time.time()-start_time))*1000)
    # print(pars)
    optimizer.set(**pars)
    try:
        optimizer.minimize(objective)
        # print(optimizer.sexpr())
        status = optimizer.check()

        if status == sat: # Optimal
            results = evaluate(optimizer.model(), model_variables, num_couriers, num_items)
            return tolist(results [0],num_items+1), max(results[1]),True
        
        elif status == unsat: # Unsat
            return [],'unsat',False
        
        else: # Not optimal
            results = evaluate(optimizer.model(), model_variables, num_couriers, num_items)

            if results:
                return tolist(results [0],num_items+1), max(results[1]),False
            else:
                return [], 'N/A',False
    except Exception as e:
        # raise(e)
        return [], 'N/A',False


def count_2d(matrix, rows,cols, value, count):
    return Sum([If(matrix[c][i] == value, 1, 0) for i in range(cols) for c in range(rows)]) == count

def count_1d(array, value, array_length=None):
    # print(array)
    if not array_length:
        array_length=len(array)
    return Sum([If(array[i] == value, 1, 0) for i in range(array_length)])

def at_least_one(array, value, array_length=None):
    if not array_length:
        array_length=len(array)
    # print(array, value)
    return Or([array[i] == value for i in range(array_length)])

def maximum(array,array_length=None):
    if not array_length:
        array_length=len(array)
    maximum = array[0]
    for i in range(array_length):
        maximum = If(array[i] > maximum, array[i], maximum)
    return maximum


def set_default_model(data, solver):
    num_couriers = data['num_couriers']
    num_items = data['num_items']
    courier_capacities = data['courier_capacities']
    item_sizes = data['item_sizes']
    distances = data['distances']
    lower_bound = data['lower_bound']
    upper_bound = data['upper_bound']
    delivers=num_items-num_couriers+1+1
    distances_z3 = [Array(f'distances_z3_{i}', IntSort(), IntSort()) for i in range(num_items+1)]
    
    loads = [Int(f'loads_{c}') for c in range(num_couriers)]
    distances_traveled = [Int(f'distances_traveled{c}') for c in range(num_couriers)]
    stops = [Array(f'stops_{c}', IntSort(), IntSort()) for c in range(num_couriers)]
    item_resp = [Int(f'item_resp_{i}') for i in range(num_items)]
    successors = Array(f'successors', IntSort(), IntSort()) 
    longest_trip = Int('longest_trip')
    
    
    # Define distances_z3
    for i in range(num_items + 1):
        for j in range(num_items + 1):
            solver.add(distances_z3[i][j] == distances[i][j])
    
    # Define upper bounds
    for c in range(num_couriers):
        solver.add(distances_traveled[c] <= upper_bound)
    # for c in range(num_couriers):
        solver.add(distances_traveled[c] > 0)
    
    # Domain for stops
    for c in range(num_couriers):
        for i in range(delivers):
            bound=num_items if i>0 else num_items-1
            solver.add(stops[c][i] <= bound)
    # for c in range(num_couriers):
    #     for i in range(num_items):
            solver.add(stops[c][i] >= 0)
    
    # Domain for item_resp
    for i in range(num_items):
        solver.add(item_resp[i] <num_couriers)
    # for i in range(num_items):
        solver.add(item_resp[i] >= 0)

    # define successors
    # for i in range(num_items):
    #     solver.add(successors[i] <= num_items)
    #     solver.add(successors[i] >= 0)

    # Define loads
    for c in range(num_couriers):
        solver.add(loads[c] > 0)
    # for c in range(num_couriers):
        solver.add(loads[c] == Sum([
            If(item_resp[i] == c, item_sizes[i], 0) for i in range(num_items)]))
    
    # === CONSTRAINTS ===
    # Bin packing constraint
    for c in range(num_couriers):
        solver.add(loads[c] <= courier_capacities[c])
    
    # each courier must be responsible for delivering at least one item
    # for c in range(num_couriers):
        solver.add(Or([item_resp[i] == c for i in range(num_items)]))

        
    #all items must be delivered
    for i in range(num_items):
        solver.add(count_2d(stops,num_couriers,delivers, i, 1))
        
    #channeling constraint
    for c in range(num_couriers):
        for i in range(delivers):
            c1 = i< count_1d(item_resp, c)
            c2 = stops[c][i] != num_items 
            solver.add(Implies(c1, c2))
            solver.add(Implies(c2, c1))
    
    # symmetry breaking constraint
    for c1 in range(num_couriers):
        for c2 in range(num_couriers):
            if c1 != c2:
                solver.add(Implies(courier_capacities[c1] > courier_capacities[c2], loads[c1] > loads[c2] ))
    # for c1 in range(num_couriers):
    #     for c2 in range(num_couriers):
    #         if c1 != c2:
                solver.add(Implies(courier_capacities[c1] < courier_capacities[c2], loads[c1] <= loads[c2] ))
    # for c1 in range(num_couriers):
    #     for c2 in range(num_couriers):
    #         solver.add(Implies(courier_capacities[c1] < courier_capacities[c2], loads[c1] < loads[c2] ))
    # all the items assigned to a courier in 'item_responsibility' must appear in the courier's route in 'stops'
    for c in range(num_couriers):
        for i in range(num_items):
            solver.add(Implies(item_resp[i] == c, at_least_one(stops[c], i, delivers)))
    
    # distance calculation
    for c in range(num_couriers):
        for i in range(delivers-1):
            solver.add(Implies(stops[c][i] != num_items, successors[stops[c][i]] == stops[c][i+1]))    
    # for c in range(num_couriers):
        solver.add(distances_traveled[c] == Sum([If(item_resp[i] == c, distances_z3[i][successors[i]],0) for i in range(num_items)])
                    + distances_z3[num_items][stops[c][0]])# + distances_z3[stops[c][delivers-1]][num_items-1]) #TODO
        
    solver.add(longest_trip == maximum(distances_traveled))
    solver.add(longest_trip >= lower_bound)
    solver.add(longest_trip <= upper_bound)
    return longest_trip, [loads, distances_traveled, stops, item_resp, successors]

    #168


