from z3 import *
import SAT.SAT_solver as SAT_solver
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import CheckButtons

def plot_solution_3d(solution, num_couriers, num_items, num_orders):
    # Create a 3D matrix initialized to 0
    matrix = np.zeros((num_couriers, num_items, num_orders), dtype=int)
    
    # Update the matrix with the solution values
    for (c, i, o) in solution:
        matrix[c][i][o] = 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = np.where(matrix == 1, 'green', 'red')
    alpha = np.where(matrix == 1, 0.6, 0.2)
    
    def draw_single_cube(ax, position, color, alpha, value):
        """ Draws a single cube at the given position with the given color and transparency. """
        r = [0, 1]
        vertices = np.array([[x, y, z] for x in r for y in r for z in r])
        vertices = vertices + np.array(position)
        
        faces = [[vertices[j] for j in [0, 1, 3, 2]],
                 [vertices[j] for j in [4, 5, 7, 6]],
                 [vertices[j] for j in [0, 1, 5, 4]],
                 [vertices[j] for j in [2, 3, 7, 6]],
                 [vertices[j] for j in [0, 2, 6, 4]],
                 [vertices[j] for j in [1, 3, 7, 5]]]
        
        poly3d = Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=0.5, alpha=alpha)
        ax.add_collection3d(poly3d)
        
        # Add text on top of the cube
        x, y, z = position
        text = ax.text(x + 0.5, y + 0.5, z + 0.5, str(value), color='black', ha='center', va='center', fontsize=8, weight='bold')
        return poly3d, text

    drawn_cubes = []

    def draw_cubes():
        while drawn_cubes:
            poly3d, text = drawn_cubes.pop()
            poly3d.remove()
            text.remove()
        for c in range(num_couriers):
            if not couriers_visible[c]:
                continue
            for i in range(num_items):
                if not items_visible[i]:
                    continue
                for o in range(num_orders):
                    drawn_cubes.append(draw_single_cube(ax, (c, i, o), colors[c, i, o], alpha[c, i, o], matrix[c, i, o]))

    couriers_visible = [True] * num_couriers
    items_visible = [True] * num_items

    draw_cubes()

    ax.set_xlabel('Courier')
    ax.set_ylabel('Item')
    ax.set_zlabel('Order')
    ax.set_xticks(range(num_couriers))
    ax.set_yticks(range(num_items))
    ax.set_zticks(range(num_orders))
    
    # Adjust the view and scaling for better visibility
    ax.view_init(elev=20., azim=-35)
    ax.set_box_aspect([num_couriers, num_items, num_orders])  # Aspect ratio is 1:1:1
    
    # Adjust the limits of the axes
    ax.set_xlim([0, num_couriers])
    ax.set_ylim([0, num_items])
    ax.set_zlim([0, num_orders])
    
    # Create widgets for interactivity
    ax_checkbox_couriers = plt.axes([0.01, 0.4, 0.15, 0.15])
    ax_checkbox_items = plt.axes([0.01, 0.2, 0.15, 0.15])
    
    courier_labels = [f'Courier {i}' for i in range(num_couriers)]
    item_labels = [f'Item {i}' for i in range(num_items)]
    
    courier_check = CheckButtons(ax_checkbox_couriers, courier_labels, couriers_visible)
    item_check = CheckButtons(ax_checkbox_items, item_labels, items_visible)
    
    def courier_func(label):
        index = courier_labels.index(label)
        couriers_visible[index] = not couriers_visible[index]
        draw_cubes()

    def item_func(label):
        index = item_labels.index(label)
        items_visible[index] = not items_visible[index]
        draw_cubes()

    courier_check.on_clicked(courier_func)
    item_check.on_clicked(item_func)

    plt.show()

def print_2D_matrix(matrix, model, ax1, ax2):
    r1 = len(matrix)
    r2 = len(matrix[0])
    print(f"The matrix is {r1}x{r2}")
    # Stampa la barra superiore degli indici
    print("\t", end="")
    for j in range(r2):
        print(f"{ax2}_{j}", end="\t")
    print()

    for i in range(r1):
        # Stampa la barra laterale degli indici
        print(f"{ax1}_{i}", end="\t")
        for j in range(r2):
            if model.evaluate(matrix[i][j]):  # If the element is True
                print(1, end="\t")  # Print 1 followed by a tab
            else:  # If the element is False
                print(0, end="\t")  # Print 0 followed by a tab
        print()  # Move to the next line after printing a row


def display_instance(instance):
    print("num_items:", instance['num_items'])
    print("courier_capacities:", instance['courier_capacities'])
    print("item_sizes:", instance['item_sizes'])
    print("distances:")
    for row in instance['distances']:
        print("  ", row)
    print("lower_bound:", instance['lower_bound'])
    print("upper_bound:", instance['upper_bound'])
    
def binary_to_int(bits, order_bits):
    return sum(If(bits[b], 2**b, 0) for b in range(order_bits))

def solve_strategy(
    instance_data, 
    model,
    strategy,
    params,
    execTime,
    binary_cut=2,
    incremental_factor=2,
    symmetry=False, # if we want to use SB
    distance_symmetry=False,
    verbose_search=False,
    verbose_solver=False
) :
    """
    Searches for a solution to a given instance using different search strategies.

    Parameters:
    - instance_data (dict): Contains information about the instance, including 'lower_bound' and 'upper_bound'.
    - model (str): The model type to use for solving. Options are:
        - "binary": Uses the binary model for solving.
        - "1-hot": Uses the 1-hot model for solving.
        - "circuit": Uses the circuit model for solving. 
    - strategy (str): The search strategy to use. Options include:
        - "lower_upper": Incrementally increases the bound from lower_bound to upper_bound.
        - "upper_lower": Decrementally decreases the bound from upper_bound to lower_bound.
        - "binary_search": Uses a binary search to find the minimum bound that yields a SAT solution.
        - "incremental_lower_upper": Increases the bound exponentially until a SAT solution is found, then searches backward to find the minimum bound.
    - params (dict): Additional parameters for the search, including 'timeout'.
    - execTime (float): The starting execution time of the search.
    - binary_cut (int, optional): The factor by which to divide the search space in "binary_search" mode. Defaults to 2.
    - incremental_factor (int, optional): The factor by which to increase the bound in "incremental_lower_upper" mode. Defaults to 2.
    - symmetry (bool, optional): If True, considers symmetry in the solving process. Defaults to False.
    - verbose_search (bool, optional): If True, prints detailed debug information about the search process. Defaults to False.
    - verbose_solver (bool, optional): If True, prints detailed debug information from the solver. Defaults to False.

    Returns:
    - obj (str or int): The objective value, either the minimum bound found for a SAT solution or 'N/A' if no solution is found.
    - solution (list): The solution corresponding to the minimum bound found.
    """
    obj='N/A'
    solution = []
    
    aux=params.copy()
    
    if verbose_search : 
        l=len(instance_data['distances'])*4
        print("#"*10, " INSTANCE ", "#"*(l-10))
        display_instance(instance_data)
        print("#"*(l+10))
    
    solve=None
    match model :
        case "1-hot-cube" : solve = SAT_solver.solve_hot
        case "binary-cube" : solve = SAT_solver.solve_bin
        case "circuit" : solve = SAT_solver.solve_circuit
    
    match strategy:
        case "LU" :
            # Mode 1: lower --> upper
            for max_path in range(instance_data['lower_bound'],instance_data['upper_bound']+1):
                # Timer
                try:
                    aux['timeout'] = params['timeout']-(time.time()-execTime)
                    if aux['timeout']<=0:
                        break
                except:pass
                
                # Execution
                result, objective, solution = solve(instance_data, max_path, aux, verbose=verbose_solver, symmetry=symmetry, distance_symmetry=distance_symmetry)

                # Backup
                if verbose_search : print(f"max_path={max_path}\tsolution={solution}")
                if result=='sat':
                    obj=objective
                    break
                
        case "UL" :
            # Mode 2: upper --> lower 
            for max_path in range(instance_data['upper_bound'],instance_data['lower_bound']-1, -1):
                # Timer
                try:
                    aux['timeout'] = params['timeout']-(time.time()-execTime)
                    if aux['timeout']<=0:
                        break
                except:pass
                
                # Execution
                result, objective, sol = solve(instance_data, max_path, aux, verbose=verbose_solver, symmetry=symmetry, distance_symmetry=distance_symmetry)
                 
                # Backup
                if verbose_search : print(f"max_path={max_path}\tsolution={solution}")
                if result=='sat' : 
                    obj=objective
                    solution = sol
                else :
                    break
            
        case "BS" :
            # Mode 3: binary search
            lower_bound = instance_data['lower_bound']
            upper_bound = instance_data['upper_bound']
            min = lower_bound
            while lower_bound <= upper_bound:
                # Timer
                try:
                    aux['timeout'] = params['timeout']-(time.time()-execTime)
                    if aux['timeout'] <= 0:
                        break
                except:
                    pass
                    
                # Execution    
                mid = (upper_bound + lower_bound) // binary_cut
                result, objective, sol = solve(instance_data, mid, aux, verbose=verbose_solver, symmetry=symmetry, distance_symmetry=distance_symmetry)

                # Backup + update value
                if result == 'sat':
                    obj = objective
                    solution = sol
                    # print(solution)
                    upper_bound = objective - 1  # Try for a smaller feasible solution
                else:
                    lower_bound = mid + 1  # Try for a larger feasible solution
                if verbose_search : print(f"\t\tmax_path={objective}\tsolution={solution}")
            
        case "ILU":
            # Mode 4: incremental lower --> upper
            lower_bound = instance_data['lower_bound']
            upper_bound = instance_data['upper_bound']
            bound = lower_bound
            while bound <= upper_bound:
                # Timer
                try:
                    aux['timeout'] = params['timeout'] - (time.time() - execTime)
                    if aux['timeout'] <= 0:
                        break
                except:
                    pass

                # Execution
                result, objective, sol = solve(instance_data, bound, aux, verbose=verbose_solver, symmetry=symmetry, distance_symmetry=distance_symmetry)
                if verbose_search : print(f"{bound}) In the incremental part i found: ", obj, sol)

                if result == 'sat':
                    # We have found a sat solution but maybe it's not the smallest
                    obj = objective
                    solution = sol
                    bound -= 1 # we make a step back to check
                    while bound >= lower_bound:
                        # Timer
                        try:
                            aux['timeout'] = params['timeout'] - (time.time() - execTime)
                            if aux['timeout'] <= 0:
                                break
                        except:
                            pass

                        # Execution
                        result, objective, sol = solve(instance_data, bound, aux, verbose=verbose_solver, symmetry=symmetry, distance_symmetry=distance_symmetry)
                        
                        # Backup
                        if result == 'sat':
                            # The previous solution wasn't the smalles, continue the search
                            obj = objective
                            solution = sol
                        else:
                            # The previous one was the smalles
                            break
                        # Update values
                        if verbose_search : print(f"{bound}) In the backtracking part i found: ", obj, sol)
                        bound -= 1
                    break  # We found the smallest or the timout ended
                else:
                    # Update values
                    # The solution is unsat so we need a bigger bound
                    next_bound = bound * incremental_factor
                    # Check not to use a too big bound
                    if next_bound > upper_bound : 
                        bound += 1 
                        if verbose_search : print(f"Unsat, try incrementing by 1 in order not to overcome {upper_bound}: bound = {bound}")
                    else : 
                        bound = next_bound
                        if verbose_search : print(f"Unsat, try incrementing by {incremental_factor}: bound = {bound}")
    return obj, solution

def break_subcircuits(solver, successor, num_items):
    # Create auxiliary variables for reachability
    reach = [[Bool(f'reach_{i}_{j}') for j in range(num_items)] for i in range(num_items)]
    
    # An item i can reach itself (base of reachability)
    for i in range(num_items):
        solver.add(reach[i][i])
        
    # If item i can reach item j and item j can reach item k, then item i can reach item k
    for i in range(num_items):
        for j in range(num_items):
            for k in range(num_items):
                if i != j and j != k and i != k:
                    # solver.add(Implies(And(reach[i][j], reach[j][k]), reach[i][k]))
                    solver.add(Or(Not(And(reach[i][j], reach[j][k])), reach[i][k]))

    # If item i has a successor j, then i can reach j
    for i in range(num_items):
        for j in range(num_items):
            # solver.add(Implies(successor[i][j], reach[i][j]))
            solver.add(Or(Not(successor[i][j]), reach[i][j]))
    
    # Prevent cycles (an item should not be able to reach itself through any other item)
    for i in range(num_items):
        for j in range(num_items):
            if i != j:
                solver.add(Not(And(reach[i][j], reach[j][i])))