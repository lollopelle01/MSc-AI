import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import parse_dzn
import z3


def define_decision_variables(data,use_arrays):
    '''
	couriers_capa,items_size,distances,stops,items_resp,distances_travld
    '''
    num_couriers = data['num_couriers']
    num_items = data['num_items']
    courier_capacities = data['courier_capacities']
    item_sizes = data['item_sizes']
    distances = data['distances']
    upper_bound = data['upper_bound']
    delivers=num_items-num_couriers+1+1
    s = ''
    if use_arrays:
        # capacities
        s += f'(declare-fun couriers_capa () (Array Int Int))\n'
        for c in range(1,num_couriers+1):
            s += f'(assert (= (select couriers_capa {c}) {courier_capacities[c-1]}))\n'
        # sizes
        s += f'(declare-fun items_size () (Array Int Int))\n'
        for i in range(1,num_items+1):
            s += f'(assert (= (select items_size {i}) {item_sizes[i-1]}))\n'
        # distances
        for i in range(1,num_items+2):
            s += f'(declare-fun distances_{i} () (Array Int Int))\n'
            for j in range(1,num_items+2):
                s += f'(assert (= (select distances_{i} {j}) {distances[i-1][j-1]}))\n'
        # stops
        for c in range(1,num_couriers+1):
            s += f'(declare-fun stops_{c} () (Array Int Int))\n'
            for i in range(1,delivers+1):
                s += f'(assert (>= (select stops_{c} {i}) 1))\n'
                s += f'(assert (<= (select stops_{c} {i}) {num_items if i==1 else num_items+1}))\n'
        # resp
        s += f'(declare-fun items_resp () (Array Int Int))\n'
        for i in range(1,num_items+1):
            s += f'(assert (>= (select items_resp {i}) 1))\n'
            s += f'(assert (<= (select items_resp {i}) {num_couriers}))\n'
        # distances traveled    
        s += f'(declare-fun distances_traveled () (Array Int Int))\n'
        for c in range(1,num_couriers+1):
            s += f'(assert (>= (select distances_traveled {c}) 0))\n'
            s += f'(assert (<= (select distances_traveled {c}) {upper_bound}))\n'
    else:
        for c in range(1,num_couriers+1):
            s += f'(declare-fun courier_{c}_capa () Int)\n'
            s += f'(assert (= courier_{c}_capa {courier_capacities[c-1]}))\n'
        for i in range(1,num_items+1):
            s += f'(declare-fun item_{i}_size () Int)\n'
            s += f'(assert (= item_{i}_size {item_sizes[i-1]}))\n'
        for i in range(1,num_items+2):
            for j in range(1,num_items+2):
                s += f'(declare-fun distance_{i}_{j} () Int)\n'
                s += f'(assert (= distance_{i}_{j} {distances[i-1][j-1]}))\n'
        for c in range(1,num_couriers+1):
            for i in range(1,delivers+1):
                s += f'(declare-fun stop_{c}_{i} () Int)\n'
                s += f'(assert (>= stop_{c}_{i} 1))\n'
                s += f'(assert (<= stop_{c}_{i} {num_items if i==1 else num_items+1}))\n'
        for i in range(1,num_items+1):
            s += f'(declare-fun item_{i}_resp () Int)\n'
            s += f'(assert (>= item_{i}_resp 1))\n'
            s += f'(assert (<= item_{i}_resp {num_couriers}))\n'
        for c in range(1,num_couriers+1):
            s += f'(declare-fun distance_{c}_traveled () Int)\n'
            s += f'(assert (>= distance_{c}_traveled 0))\n'
            s += f'(assert (<= distance_{c}_traveled {upper_bound}))\n'
    return s

def define_bin_packing(nc,ni,arrays):
    s=''
    if arrays:
        s+=f'(declare-fun loads () (Array Int Int))\n'
        for c in range(1,nc+1):
            s+=f'(assert (= (select loads {c}) (+ '
            for i in range(1,ni+1):
                s+=f'(ite (= (select items_resp {i}) {c}) (select items_size {i}) 0)'
            s+=')))\n'
            s+=f'(assert (>= (select couriers_capa {c}) (select loads {c})))\n'
    else:
        for c in range(1,nc+1):
            s+=f'(declare-fun load_{c} () Int)\n'
            s+=f'(assert (= load_{c} (+'
            for i in range(1,ni+1):
                s+=f'(ite (= item_{i}_resp {c}) item_{i}_size 0)'
            s+=')))\n'
            s+=f'(assert (>= courier_{c}_capa load_{c}))\n'
    return s

def define_couriers_duty(nc,ni,arrays):
    s=''
    if arrays:
        for c in range(1,nc+1):
            s+=f'(assert (or '
            for i in range(1,ni+1):
                s+=f'(= (select items_resp {i}) {c})'
            s+='))\n'
    else:
        for c in range(1,nc+1):
            s+=f'(assert (or '
            for i in range(1,ni+1):
                s+=f'(= item_{i}_resp {c})'
            s+='))\n'
    return s


def define_deliver_everything(nc,ni,arrays):
    s=''
    if arrays:
        for i in range(1,ni+1):
            s+=f'(assert (= (+'
            for c in range(1,nc+1):
                for j in range(1,ni-nc+2):
                    s+=f'(ite (= (select stops_{c} {j}) {i}) 1 0) '
            s+=f') 1))\n'
    else:
        for i in range(1,ni+1):
            s+=f'(assert (= (+'
            for c in range(1,nc+1):
                for j in range(1,ni-nc+2):
                    s+=f'(ite (= stop_{c}_{j} {i}) 1 0) '
            s+=f') 1))\n'
    return s

def define_padding(nc,ni,arrays):
    s=''
    if arrays:
        for c in range(1,nc+1):
            for i in range(1,ni-nc+3):
                lhs=f'(<= {i} (+ '
                for j in range(1,ni+1):
                    lhs+=f'(ite (= (select items_resp {j}) {c}) 1 0) '
                lhs+=f'))'
                rhs=f'(< (select stops_{c} {i}) {ni+1})'
                # s+=f'(assert (let ((lhs {lhs})\n'
                # s+=f'\t\t\t (rhs {rhs}))\n'
                # s+=f'\t\t\t (and (=> lhs rhs) (=> rhs lhs))))\n'
                s+=f'(assert (=> {lhs} {rhs}))\n'
                s+=f'(assert (=> {rhs} {lhs}))\n'
    else:
        for c in range(1,nc+1):
            for i in range(1,ni-nc+3):
                lhs=f'(<= {i} (+ '
                for j in range(1,ni+1):
                    lhs+=f'(ite (= item_{j}_resp {c}) 1 0) '
                lhs+=f'))'
                rhs=f'(< stop_{c}_{i} {ni+1})'
                # s+=f'(assert (let ((lhs {lhs})\n'
                # s+=f'\t\t\t (rhs {rhs}))\n'
                # s+=f'\t\t\t (and (=> lhs rhs) (=> rhs lhs))))\n'
                s+=f'(assert (=> {lhs} {rhs}))\n'
                s+=f'(assert (=> {rhs} {lhs}))\n'
    return s

def define_courier_items(nc,ni,arrays):
    s=''
    if arrays:
        for c in range(1,nc+1):
            for i in range(1,ni+1):
                s+=f'(assert (=> (= (select items_resp {i}) {c}) (or '
                for j in range(1,ni-nc+3):
                    s+=f'(= (select stops_{c} {j}) {i})'
                s+=f')))\n'
    else:
        for c in range(1,nc+1):
            for i in range(1,ni+1):
                s+=f'(assert (=> (= item_{i}_resp {c}) (or '
                for j in range(1,ni-nc+3):
                    s+=f'(= stop_{c}_{j} {i})'
                s+=f')))\n'
    return s

def define_successors_distances(nc,ni,lowerbound,arrays):
    s,d='',''
    if arrays:
        s+=f'(declare-const successors (Array Int Int))\n'
        # for i in range(1,ni+1):
        #     s+=f'(assert (> (select successors {i}) 0))\n' if lowerbound else ""
        #     s+=f'(assert (<= (select successors {i}) {ni+1}))\n'
        for c in range(1,nc+1):
            for i in range(1,ni-nc+2):
                A=f'(not (= (select stops_{c} {i}) {ni+1}))'
                B=f'(= (select successors (select stops_{c} {i})) (select stops_{c} {i+1}))'
                s+=f'(assert (=> {A} {B}))\n'
        
        for c in range(1,nc+1):
            d+=f'(assert (= (select distances_traveled {c}) (+ '
            for i in range(1,ni+1):
                for j in range(1,ni+2):
                    d+=f'(ite (and (= (select items_resp {i}) {c}) (= (select successors {i}) {j}) ) (select distances_{i} {j}) 0)'
            d+=f'(select distances_{ni+1} (select stops_{c} 1))'
            d+=f')))\n'
    else:
        for i in range(1,ni+1):
            s+=f'(declare-fun successor_of_{i} () Int)\n'
            # s+=f'(assert (> successor_of_{i} 0))\n' if lowerbound else ""
            # s+=f'(assert (<= successor_of_{i} {ni+1}))\n'

        for c in range(1,nc+1):
            for i in range(1,ni-nc+2):
                for j in range(1,ni+1):
                    # logically a->(b->c) is not(a and b and not(c))
                    A=f'(not (= stop_{c}_{i} {ni+1}))'
                    B=f'(= stop_{c}_{i} {j})'
                    C=f'(= successor_of_{j} stop_{c}_{i+1})'
                    s+=f'(assert (not (and {A} {B} (not {C}))))\n'

        for c in range(1,nc+1):
            d+=f'(assert (= distance_{c}_traveled (+ '
            for i in range(1,ni+1):
                for j in range(1,ni+2):
                    d+=f'(ite (and (= item_{i}_resp {c}) (= successor_of_{i} {j}) ) distance_{i}_{j} 0)'
            for j in range(1,ni+2):
                d+=f'(ite (= {j} stop_{c}_1) distance_{ni+1}_{j} 0)'
            d+=f')))\n'
    return s,d

def define_distances(nc,ni,arrays):
    raise TypeError('non implementato senza successors')
    s=''
    if arrays:
        for c in range(1,nc+1):
            s+=f'(assert (= (select distances_traveled {c}) (+ '
            for i in range(1,ni):
                for j in range(1,ni+2):
                    s+=f'(ite (= (select stops_{c} {i}) {j}) (select distances_{j} (select stops_{c} {i+1})) 0)'
            s+=f'(select distances_{ni+1} (select stops_{c} 1))'
            s+=f')))\n'
    else:
        for c in range(1,nc+1):
            s+=f'(assert (= distance_{c}_traveled (+ '
            for i in range(1,ni):
                for j in range(1,ni+2):
                    for k in range(1,ni+2):
                        s+=f'(ite (and (= stop_{c}_{i} {j})(= stop_{c}_{i+1} {k})) distance_{j}_{k} 0)'
            for j in range(1,ni+1):
                s+=f'(ite (= {j} stop_{c}_1) distance_{ni+1}_{j} 0)'
            s+=f')))\n'
    return s

def define_simmetry(nc,method,arrays):
    s=''
    for c1 in range(1,nc+1):
        for c2 in range(1,nc+1):
            if c1!=c2:
                match method:
                    case 'both':
                        if arrays:
                            s+=f'(assert (=> (> (select couriers_capa {c1}) (select couriers_capa {c2})) (> (select loads {c1}) (select loads {c2}))))\n'
                            s+=f'(assert (=> (< (select couriers_capa {c1}) (select couriers_capa {c2})) (< (select loads {c1}) (select loads {c2}))))\n'
                        else:
                            s+=f'(assert (=> (> courier_{c1}_capa courier_{c2}_capa) (> load_{c1} load_{c2})))\n'
                            s+=f'(assert (=> (< courier_{c1}_capa courier_{c2}_capa) (< load_{c1} load_{c2})))\n'
                    case '>':
                        if arrays:
                            s+=f'(assert (=> (> (select couriers_capa {c1}) (select couriers_capa {c2})) (> (select loads {c1}) (select loads {c2}))))\n'
                        else:
                            s+=f'(assert (=> (> courier_{c1}_capa courier_{c2}_capa) (> load_{c1} load_{c2})))\n'
                    case '<':
                        if arrays:
                            s+=f'(assert (=> (< (select couriers_capa {c1}) (select couriers_capa {c2})) (< (select loads {c1}) (select loads {c2}))))\n'
                        else:
                            s+=f'(assert (=> (< courier_{c1}_capa courier_{c2}_capa) (< load_{c1} load_{c2})))\n'
                    case 'courier-item_order':
                        if c2<=c1 or c1==nc:
                            continue
                        if arrays:
                            s+=f'(assert (=> (= (select couriers_capa {c1}) (select couriers_capa {c2})) (< (select stops_{c1} 1) (select stops_{c2} 1))))\n'
                        else:
                            s+=f'(assert (=> (= courier_{c1}_capa courier_{c2}_capa) (< stop_{c1}_1 stop_{c2}_1)))\n'
                    case _: pass
    return s

def add_objective(num_couriers,obj,head,tail,arrays=True,impose_lower_bound=True):
    objective=''
    for c in range(1,num_couriers+1):
        if arrays:
            objective+=f'(assert (> (select distances_traveled {c}) 0))\n' if impose_lower_bound else ''
            objective+=f'(assert (<= (select distances_traveled {c}) {obj}))\n'
        else:
            objective+=f'(assert (> distance_{c}_traveled 0))\n' if impose_lower_bound else ''
            objective+=f'(assert (<= distance_{c}_traveled {obj}))\n'
    return head+objective+tail

def parse_solution(model,arrays,solver,best=False):
    if model==None:
        return []
    num_c=num_i=None
    match solver:
        case 'z3':
            num_c=int(str(model[[var for var in model if 'num_couriers' in var.name()][0]]))
            num_i=int(str(model[[var for var in model if 'num_items' in var.name()][0]]))

        case 'cvc5':
            solver_,vars=model            
            num_c=solver_.getValue(vars[0]).toPythonObj()
            num_i=solver_.getValue(vars[1]).toPythonObj()   
    return get_itineraries(model,arrays,solver,num_c,num_i,best)


def get_itineraries(model,arrays,solver,num_c,num_i,best):
    stops=get_stops(model,False,arrays,num_c,num_i,solver,best,print_=False)
    # print(stops)
    stops=[[int(str(e))for e in row]for row in stops]
    default=num_i+1
    stops=[[e for e in row if e!=default]for row in stops ]
    return stops

def get_variables(model,print_names=False,arrays=True,successor=True,best=False):
    num_c=int(str(model[[var for var in model if 'num_couriers' in var.name()][0]]))
    num_i=int(str(model[[var for var in model if 'num_items' in var.name()][0]]))

    get_responsabilities(model,print_names,arrays,num_i)
    get_loads(model,print_names,arrays,num_c)
    get_stops(model,print_names,arrays,num_c,num_i)
    if successor:get_successor(model,print_names,arrays,num_i,best)
    get_distances(model,print_names,arrays,num_c)

def get_responsabilities(model,print_names,arrays,num_i):
    variables = []
    if arrays:
        var=[model[var] for var in model if 'items_resp' in var.name()][0]
        print('resp:',convertArrayRef(var,num_i))
    else:
        for var in model:
            if '_resp' in var.name():
                variables.append((str(var).split('_resp')[0].split('item_')[1], model[var]))
        sorted_variables = sorted(variables, key=lambda x: int(x[0]))
        print('resp:',[v[1] for v in sorted_variables],[v[0] for v in sorted_variables] if print_names else '')

def get_loads(model,print_names,arrays,num_c):
    variables = []
    if arrays:
        var=[model[var] for var in model if 'loads' in var.name()][0]
        print('loads:',convertArrayRef(var,num_c))
    else:
        for var in model:
            if 'load_' in var.name():
                variables.append((str(var).split('load_')[1], model[var]))
        sorted_variables = sorted(variables, key=lambda x: int(x[0]))
        print('loads:',[v[1] for v in sorted_variables],[v[0] for v in sorted_variables] if print_names else '')

def get_stops(model,print_names,arrays,num_c,num_i,solver,best,print_=True):
    match solver:
        case 'z3':
            matrix=[]
            matrix_names=[]
            if arrays:
                for i in range(1,num_c+1):
                    row=[var for var in model if f'stops_{i}'==var.name()][0]
                    # print(model[row])
                    matrix.append(convertArrayRef(model[row],num_i-num_c+2))
                if print_:
                    print('stops:')
                    for i in range(len(matrix)):
                        print('  ',matrix[i])
                # return matrix
            else:
                for i in range(1,num_c+1):
                    matrix.append([])
                    matrix_names.append([])
                    for j in range(1,num_i-num_c+3):
                        node=[var for var in model if f'stop_{i}_{j}'==var.name()][0]
                        matrix[i-1].append(model[node])
                        matrix_names[i-1].append(node)
                if print_:
                    print('stops:')
                    for i in range(len(matrix)):
                        print('  ',matrix[i],matrix_names[i] if print_names else '')
            return matrix
        case 'cvc5':
            solver,vars=model
            stops=[]
            if arrays:
                stops=list(solver.getValue(vars[-(4+num_c):-4]))
                stops=[dict2array(s.toPythonObj()) for s in stops]
        
            elif best:
                s=5+num_c+num_i*2
                # print(vars[s:s+num_c*(num_i-num_c+2)])
                stops_=list(solver.getValue(vars[s:s+num_c*(num_i-num_c+2)]))
                for c in range(num_c):
                    stops.append([])
                    for i in range(num_i-num_c+2):
                        stops[c].append(stops_[c*(num_i-num_c+2)+i])
            else:
                s=4+num_c+num_i+(num_i+1)**2
                # print(vars[s:s+num_c*(num_i-num_c+2)])
                stops_=list(solver.getValue(vars[s:s+num_c*(num_i-num_c+2)]))
                for c in range(num_c):
                    stops.append([])
                    for i in range(num_i-num_c+2):
                        stops[c].append(stops_[c*(num_i-num_c+2)+i])
            return stops

def get_successor(model,print_names,arrays,num_i,best_model=False):
    variables = []
    if arrays or best_model:
        var=[model[var] for var in model if 'successors' in var.name()][0]
        print('succ:',convertArrayRef(var,num_i))
    else:
        for var in model:
            if 'successor' in var.name():
                variables.append((str(var).split('successor_of_')[1], model[var]))
        sorted_variables = sorted(variables, key=lambda x: int(x[0]))
        print('succ:',[v[1] for v in sorted_variables],[v[0] for v in sorted_variables] if print_names else '')    

def get_distances(model,lib,print_names,arrays,best=False,num_c=None,print_=True):
    variables = []
    match lib:
        case 'z3':
            if arrays:
                var=[model[var] for var in model if 'distances_traveled' in var.name()][0]
                variables=convertArrayRef(var,num_c)
                if print_:print('dist:',variables)
                return variables
            else:
                for var in model:
                    if '_traveled' in var.name():
                        variables.append((str(var).split('_traveled')[0].split('distance_')[1], model[var]))
                sorted_variables = sorted(variables, key=lambda x: int(x[0]))
                if print_:print('dists:',[v[1] for v in sorted_variables],[v[0] for v in sorted_variables] if print_names else '')
                return [int(str(v[1])) for v in sorted_variables]
        case 'cvc5':
            if arrays:
                # print('arrays non best')
                solver,vars=model
                distances=solver.getValue(vars[-3]).toPythonObj()
                return list(distances.values())
            else:
                if best:
                    # print('best')
                    solver,vars=model
                    num_c=solver.getValue(vars[0]).toPythonObj()
                    distances=list(solver.getValue(vars[-(num_c+1+num_c):-(num_c+1)]))
                    # print([d.toPythonObj() for d in distances])
                    return [d.toPythonObj() for d in distances]
                else:
                    # print('non arrays')
                    solver,vars=model
                    num_c=solver.getValue(vars[0]).toPythonObj()
                    num_i=solver.getValue(vars[1]).toPythonObj()
                    distances=list(solver.getValue(vars[-(num_c+num_c+num_i):-(num_c+num_i)]))
                    # print([d.toPythonObj() for d in distances])
                    return [d.toPythonObj() for d in distances]


    
def check_model_params(input_dict):
    model_params={
        'simmetry_method':'>',
        'use_successors':True,
        'use_arrays':True,
        'impose_lower_bound':False,
        'redundancy':True,
        'best':False,
        'z3':False,
    }

    if not input_dict:
        return model_params

    expected_types = {
        'simmetry_method': str,
        'use_successors': bool,
        'use_arrays': bool,
        'impose_lower_bound': bool,
        'redundancy': bool,
        'best':bool,
        'z3':bool,
    }

    for key in input_dict.keys():
        if key not in model_params:
            raise KeyError(f"Unexpected key: {key}")
        if not isinstance(input_dict[key], expected_types[key]):
            raise TypeError(f"Incorrect type for key: {key}. Expected {expected_types[key].__name__}, got {type(input_dict[key]).__name__}.")
    validated_dict = model_params.copy()
    validated_dict.update(input_dict)
    
    return validated_dict

def dict2array(dic):
    max_index = max(dic.keys())
    default=dic['cdmo']
    dic.pop('cdmo',None)
    result_list = [default] * (max_index)
    for index, value in dic.items():
        result_list[index-1] = value
    return result_list

def convertArrayRef(array,length=None):
    try:
        s=str(array).replace('\n','').replace(' ','')
        _,s=s.split('K(Int,')
        s=s.split(')')[:-1]
        default=int(s[0])
        s=s[1:]
        pairs=[]
        for pair in s:
            vals=pair.split(',')
            pairs.append((int(vals[1]),int(vals[2])))
        res=[0]
        size=0
        if length:
            size=length if length>size else size
        if pairs:
            aux=max([i[0] for i in pairs])
            size=aux if aux>size else size
        res=[default]*size
        # print(size)
        # print(pairs,size,res)
        for pair in pairs:
            res[pair[0]-1]=pair[1]
    except IndexError as e:
        print('!!exception!!',array,e)
        raise e
        return [0]
    return res



if __name__=='__main__':
    array='''K(Int, 14)'''
    converted=convertArrayRef(array,5)
    print(converted)

def generate_best_model(instance_data,sym):
    # instance_data=parse_dzn(dzn_instance)

    num_couriers = instance_data['num_couriers']
    num_items = instance_data['num_items']
    courier_capacities = instance_data['courier_capacities']
    item_sizes = instance_data['item_sizes']
    distances = instance_data['distances']
    lower_bound = instance_data['lower_bound']
    upper_bound = instance_data['upper_bound']
    delivers=num_items-num_couriers+1+1
    # ===== INSTANCE VARIABLES =====
    model_h = f'''
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
    for c, size in enumerate(courier_capacities, start=1):
        model_h += f'(declare-fun courier_{c}_capa () Int)\n'
        model_h += f'(assert (= courier_{c}_capa {size}))\n'
    for i, size in enumerate(item_sizes, start=1):
        model_h += f'(declare-fun item_{i}_size () Int)\n'
        model_h += f'(assert (= item_{i}_size {size}))\n'
    for i in range(1,num_items+2):
        model_h += f'(declare-fun distances_{i} () (Array Int Int))\n'
        for j in range(1,num_items+2):
            model_h += f'(assert (= (select distances_{i} {j}) {distances[i-1][j-1]}))\n'

    # ===== DECISION VARIABLES =====
    for c in range(1,num_couriers+1):
        for i in range(1,delivers+1):
            model_h += f'(declare-fun stop_{c}_{i} () Int)\n'
            model_h += f'(assert (>= stop_{c}_{i} 1))\n'
            model_h += f'(assert (<= stop_{c}_{i} {num_items if i==1 else num_items+1}))\n'
    for i in range(1,num_items+1):
        model_h += f'(declare-fun item_{i}_resp () Int)\n'
        model_h += f'(assert (>= item_{i}_resp 1))\n'
        model_h += f'(assert (<= item_{i}_resp {num_couriers}))\n'
    for c in range(1,num_couriers+1):
        model_h += f'(declare-fun distance_{c}_traveled () Int)\n'
        model_h += f'(assert (>= distance_{c}_traveled 0))\n'
        model_h += f'(assert (<= distance_{c}_traveled {upper_bound}))\n'
    
#     model_h += f'''
# (declare-fun longest_trip () Int)
# (assert (>= longest_trip {lower_bound}))
# (assert (<= longest_trip {upper_bound}))
# '''.lstrip()

    # ===== CONSTRAINTS =====
    # bin packing
    # -- declare and calculate couriers load
    # -- loads must not exceed capacities
    for c in range(1,num_couriers+1):
        model_h+=f'(declare-fun load_{c} () Int)\n'
        model_h+=f'(assert (= load_{c} (+'
        for i in range(1,num_items+1):
            model_h+=f'(ite (= item_{i}_resp {c}) item_{i}_size 0)'
        model_h+=')))\n'
        model_h+=f'(assert (>= courier_{c}_capa load_{c}))\n'

    # start and end in depot
    # for c in range(1,num_couriers+1):
    #     model_h += f'(declare-fun stop_{c}_1 () Int)\n'
    #     model_h += f'(assert (= stop_{c}_1 {num_items+1}))\n'
    #     model_h += f'(declare-fun stop_{c}_{num_items+2} () Int)\n'
    #     model_h += f'(assert (= stop_{c}_{num_items+2} {num_items+1}))\n'

    # each courier appears on item resp
    for c in range(1,num_couriers+1):
        model_h+=f'(assert (or '
        for i in range(1,num_items+1):
            model_h+=f'(= item_{i}_resp {c})'
        model_h+='))\n'

    # each couriers must deliver something on 1st stop
    # for c in range(1,num_couriers+1):
    #     model_h += f'(declare-fun stop_{c}_1 () Int)\n'
    #     model_h += f'(assert (>= stop_{c}_1 1))\n'
    #     model_h += f'(assert (<= stop_{c}_1 {num_items}))\n'

    # constraint the capacities to be in some sort of order
    # simmetry
    for c1 in range(1,num_couriers+1):
        for c2 in range(1,num_couriers+1):
            if c1!=c2:
                match sym:
                    case '>':
                        model_h+=f'(assert (=> (> courier_{c1}_capa courier_{c2}_capa) (> load_{c1} load_{c2})))\n'
                # model_h+=f'(assert (=> (< courier_{c1}_capa courier_{c2}_capa) (< load_{c1} load_{c2})))\n'
                    case 'courier-item_order':
                        if c2<=c1 or c1==num_couriers:
                            continue
                        model_h+=f'(assert (=> (= courier_{c1}_capa courier_{c2}_capa) (< stop_{c1}_1 stop_{c2}_1)))\n'
                    case _: raise ValueError('sym not valid for best model')
    # all items must be delivered. ===era LENTO forse mo funziona però===
    # for i in range(1,num_items+1):
    #     model_h+=f'(assert (= (+'
    #     for c in range(1,num_couriers+1):
    #         for j in range(1,num_items+1):
    #             model_h+=f'(ite (= stop_{c}_{j} {i}) 1 0) '
    #     model_h+=f') 1))'

    # channeling; stops after completing the delivers must be depot
    for c in range(1,num_couriers+1):
        for i in range(1,delivers+1):
            # lhs=>rhs
            # rhs=>lhs
            lhs=f'(<= {i} (+ '
            for j in range(1,num_items+1):
                lhs+=f'(ite (= item_{j}_resp {c}) 1 0) '
            lhs+=f'))'
            rhs=f'(< stop_{c}_{i} {num_items+1})'
            model_h+=f'(assert (=> {lhs} {rhs}))'
            model_h+=f'(assert (=> {rhs} {lhs}))'
            # model_h+=f'(assert (let ((lhs {lhs})\n'
            # model_h+=f'\t\t\t (rhs {rhs}))\n'
            # model_h+=f'\t\t\t (and (=> lhs rhs) (=> rhs lhs))))\n'

    # items delivered by c must appear in its row
    for c in range(1,num_couriers+1):
        for i in range(1,num_items+1):
            model_h+=f'(assert (=> (= item_{i}_resp {c}) (or '
            for j in range(1,delivers+1):
                model_h+=f'(= stop_{c}_{j} {i})'
            model_h+=f')))\n'

    # define successors
    model_h+=f'(declare-const successors (Array Int Int))\n'
    # model_h+=f'(assert (= (select successors 0) {num_couriers}))\n'
    # channeling of successors
    for c in range(1,num_couriers+1):
        for i in range(1,delivers):
            A=f'(not (= stop_{c}_{i} {num_items+1}))'
            B=f'(= (select successors stop_{c}_{i}) stop_{c}_{i+1})'
            model_h+=f'(assert (=> {A} {B}))\n'
    
    # compute distances with successor
    for c in range(1,num_couriers+1):
        model_h+=f'(assert (= distance_{c}_traveled (+ '
        for i in range(1,num_items+1):
            # for j in range(1,num_items+2):
                model_h+=f'(ite (= item_{i}_resp {c}) (select distances_{i} (select successors {i})) 0)'
        # for j in range(1,num_items+2):
            # model_h+=f'(ite (= {j} stop_{c}_1) distance_{num_items+1}_{j} 0)'
            # model_h+=f'(ite (= {j} stop_{c}_{num_items}) distance_{j}_{num_items} 0)'
        model_h+=f'(select distances_{num_items+1} stop_{c}_1)'
        model_h+=f')))\n'
        
    model_t='(check-sat)\n(get-model)\n'

    return model_h,model_t


def generate_smt2_model(instance_data):
    # instance_data=parse_dzn(dzn_instance)

    num_couriers = instance_data['num_couriers']
    num_items = instance_data['num_items']
    courier_capacities = instance_data['courier_capacities']
    item_sizes = instance_data['item_sizes']
    distances = instance_data['distances']
    lower_bound = instance_data['lower_bound']
    upper_bound = instance_data['upper_bound']

    # ===== INSTANCE VARIABLES =====
    model_h = f'''
(set-logic QF_LIA)
(declare-fun num_couriers () Int)
(assert (= num_couriers {num_couriers}))
(declare-fun num_items () Int)
(assert (= num_items {num_items}))
(declare-fun lower_bound () Int)
(assert (= lower_bound {lower_bound}))
(declare-fun upper_bound () Int)
(assert (= upper_bound {upper_bound}))
'''.lstrip()
    for c, size in enumerate(courier_capacities, start=1):
        model_h += f'(declare-fun courier_{c}_capa () Int)\n'
        model_h += f'(assert (= courier_{c}_capa {size}))\n'
    for i, size in enumerate(item_sizes, start=1):
        model_h += f'(declare-fun item_{i}_size () Int)\n'
        model_h += f'(assert (= item_{i}_size {size}))\n'
    for i in range(num_items+1):
        for j in range(num_items+1):
            model_h += f'(declare-fun distance_{i+1}_{j+1} () Int)\n'
            model_h += f'(assert (= distance_{i+1}_{j+1} {distances[i][j]}))\n'

    # ===== DECISION VARIABLES =====
    for c in range(1,num_couriers+1):
        for i in range(3,num_items+2):
            model_h += f'(declare-fun stop_{c}_{i} () Int)\n'
            model_h += f'(assert (>= stop_{c}_{i} 1))\n'
            model_h += f'(assert (<= stop_{c}_{i} {num_items+1}))\n'
    for i in range(1,num_items+1):
        model_h += f'(declare-fun item_{i}_resp () Int)\n'
        model_h += f'(assert (>= item_{i}_resp 1))\n'
        model_h += f'(assert (<= item_{i}_resp {num_couriers}))\n'
    for c in range(1,num_couriers+1):
        model_h += f'(declare-fun distance_{c}_traveled () Int)\n'
        model_h += f'(assert (>= distance_{c}_traveled 0))\n'
        model_h += f'(assert (<= distance_{c}_traveled {upper_bound}))\n'
    
#     model_h += f'''
# (declare-fun longest_trip () Int)
# (assert (>= longest_trip {lower_bound}))
# (assert (<= longest_trip {upper_bound}))
# '''.lstrip()

    # ===== CONSTRAINTS =====
    # bin packing
    # -- declare and calculate couriers load
    # -- loads must not exceed capacities
    for c in range(1,num_couriers+1):
        model_h+=f'(declare-fun load_{c} () Int)\n'
        model_h+=f'(assert (= load_{c} (+'
        for i in range(1,num_items+1):
            model_h+=f'(ite (= item_{i}_resp {c}) item_{i}_size 0)'
        model_h+=')))\n'
        model_h+=f'(assert (>= courier_{c}_capa load_{c}))\n'

    # start and end in depot
    for c in range(1,num_couriers+1):
        model_h += f'(declare-fun stop_{c}_1 () Int)\n'
        model_h += f'(declare-fun stop_{c}_{num_items+2} () Int)\n'
        model_h += f'(assert (= stop_{c}_1 {num_items+1}))\n'
        model_h += f'(assert (= stop_{c}_{num_items+2} {num_items+1}))\n'

    # each courier appears on item resp
    for c in range(1,num_couriers+1):
        model_h+=f'(assert (or '
        for i in range(1,num_items+1):
            model_h+=f'(= item_{i}_resp {c})'
        model_h+='))\n'

    # each couriers must deliver something on 1st stop
    for c in range(1,num_couriers+1):
        model_h += f'(declare-fun stop_{c}_2 () Int)\n'
        model_h += f'(assert (>= stop_{c}_2 1))\n'
        model_h += f'(assert (<= stop_{c}_2 {num_items}))\n'

    # constraint the capacities to be in some sort of order
    # simmetry
    for c1 in range(1,num_couriers+1):
        for c2 in range(1,num_couriers+1):
            if c1!=c2: # only the first is better than second and both
                model_h+=f'(assert (=> (> courier_{c1}_capa courier_{c2}_capa) (> load_{c1} load_{c2})))\n'
                # model_h+=f'(assert (=> (< courier_{c1}_capa courier_{c2}_capa) (< load_{c1} load_{c2})))\n'

    # all items must be delivered. ===era LENTO forse mo funziona però===
    for i in range(1,num_items+1):
        model_h+=f'(assert (= (+'
        for c in range(1,num_couriers+1):
            for j in range(2,num_items+2):
                model_h+=f'(ite (= stop_{c}_{j} {i}) 1 0) '
        model_h+=f') 1))'

    # channeling; stops after completing the delivers must be depot
    for c in range(1,num_couriers+1):
        for i in range(1,num_items+1):
            # lhs=>rhs
            # rhs=>lhs
            lhs=f'(<= {i} (+ '
            for j in range(1,num_items+1):
                lhs+=f'(ite (= item_{j}_resp {c}) 1 0) '
            lhs+=f'))'
            rhs=f'(< stop_{c}_{i+1} {num_items+1})'
            model_h+=f'(assert (=> {lhs} {rhs}))\n'
            model_h+=f'(assert (=> {rhs} {lhs}))\n'

    # items delivered by c must appear in its row
    for c in range(1,num_couriers+1):
        for i in range(1,num_items+1):
            model_h+=f'(assert (=> (= item_{i}_resp {c}) (or '
            for j in range(1,num_items+1):
                model_h+=f'(= stop_{c}_{j+1} {i})'
            model_h+=f')))\n'

    # define successors
    for i in range(1,num_items+1):
        model_h+=f'(declare-fun successor_of_{i} () Int)\n'
        # model_h+=f'(assert (> successor_of_{i} 0))\n'
        model_h+=f'(assert (<= successor_of_{i} {num_items+1}))\n'

    # channeling of successors
    for c in range(1,num_couriers+1):
        for i in range(2,num_items+2):
            for j in range(1,num_items+1):
                # logically a->(b->c) is not(a and b and not(c))
                A=f'(not (= stop_{c}_{i} {num_items+1}))'
                B=f'(= stop_{c}_{i} {j})'
                C=f'(= successor_of_{j} stop_{c}_{i+1})'
                model_h+=f'(assert (not (and {A} {B} (not {C}))))\n'

    # compute distances with successor
    for c in range(1,num_couriers+1):
        model_h+=f'(assert (= distance_{c}_traveled (+ '
        for i in range(1,num_items+1):
            for j in range(1,num_items+2):
                model_h+=f'(ite (and (= item_{i}_resp {c}) (= successor_of_{i} {j}) ) distance_{i}_{j} 0)'
        for j in range(1,num_items+2):
            model_h+=f'(ite (= {j} stop_{c}_2) distance_{num_items+1}_{j} 0)'
        model_h+=f')))\n'

    # compute distances. === ci mette 10 secondi===
    # for c in range(1,num_couriers+1):
    #     model_h+=f'(assert (= distance_{c}_traveled (+ '
    #     for i in range(1,num_items+2):
    #         for j in range(1,num_items+2):
    #             for k in range(1,num_items+2):
    #                 model_h+=f'(ite (and (= stop_{c}_{i} {j})(= stop_{c}_{i+1} {k})) distance_{j}_{k} 0)'
    #     model_h+=f')))\n'
        
    model_t='(check-sat)\n(get-model)\n'

    return model_h,model_t
