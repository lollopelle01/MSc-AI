import os
import time

def insert_commas(string:str):
    elements = string.split()
    formatted_string = ", ".join(elements)
    return formatted_string

def matrix_parser(string_matrix):
    rows = [row.strip().split() for row in string_matrix]
    
    matrix = [[] for _ in range(len(rows))]
    
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            matrix[i].append(int(rows[i][j]))
    
    return matrix

def find_upper_bound(distances,num_courier:int):
    '''
	the idea is to assign to every courier at least one item.
    Then one courier carries all the remaining items, in particular
    the furthest.
    '''
    maximums=sorted(max(row) for row in distances)[num_courier-1:]

    return sum(maximums)

def find_lower_bound(data):
    max=data[-1][0]+data[0][-1]
    for i in range(1,len(data)):
        aux=data[-1][i]+data[i][-1]
        if aux>max:
            max=aux
    return max

def find_min_load(data:str):
    loads=[int(e) for e in data.split()]
    return min(loads)

def write_new_content(data):
    content = ''
    bounds = matrix_parser(data[4:]) 
    
    content += 'num_couriers='+data[0]+';\n'
    content += 'num_items='+data[1]+';\n'
    content += 'courier_capacities=['+insert_commas(data[2])+'];\n'
    content += 'item_sizes=['+insert_commas(data[3])+'];\n'
    content += 'distances=['
    for table_row in data[4:]:
        content+='|'+insert_commas(table_row)+'\n'
    content=content[:-1]
    content+='|];\n'
    content += 'lower_bound='+str(find_lower_bound(bounds))+';\n'
    content += 'upper_bound='+str(find_upper_bound(bounds,int(data[0])))+';\n'
    return content

def read_file(path):
    result = []
    try:
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                result.append(line)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except IOError:
        print(f"Error: Unable to read file '{path}'.")
    return result


def save_dzn_file(filename, content):
    with open(filename+'.dzn', 'w') as file:
        file.write(content)

def main(argv):
    PRINT_TIME=True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        instances_folder = argv[1]
        DZN_instances_folder = argv[2]
    except:
        instances_folder = "instances"
        DZN_instances_folder = "instances_dzn"
    os.makedirs(DZN_instances_folder, exist_ok=True)
    file_names = os.listdir(instances_folder)
        
    for file_name in file_names:
        if file_name.endswith(".dat"):
            file_path = os.path.join(instances_folder, file_name)

            data=read_file(file_path)
            if PRINT_TIME:start=time.time()
            formatted_data=write_new_content(data)
            if PRINT_TIME:print(time.time()-start)

            dzn_file_name=DZN_instances_folder+'/'+file_name[:-4]
            save_dzn_file(dzn_file_name,formatted_data)
 
if __name__=='__main__':
    import sys
    main(sys.argv)