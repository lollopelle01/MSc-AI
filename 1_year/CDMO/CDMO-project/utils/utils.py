import os
import json
import check_solution as check


# SOLUTION PARSING
def tolist(solution,depot):
    '''
	solution is a string [[i1,i2],[i3,i4,i5],...[]]

    output must be that string parsed to a list
    '''
    res = []
    for r in solution:
        r = list(dict.fromkeys(r))
        if depot in r:
            r.remove(depot)
        res.append(r)
    return res

# JSONS
class CustomJSONEncoder(json.JSONEncoder):
    def write_JSON_from_dict(self, dict,_one_shot):
        if self.indent is None:
            return super().iterencode(dict,_one_shot)
        # return super().iterencode(dict,_one_shot)
        s='{\n'

        for method in dict:
            s+='\t\"'+f'{method}'+'":{\n'
            s+='\t\t"time":'+f'{dict[method]["time"]},\n'
            s+='\t\t"optimal":'+f'{"true" if dict[method]["optimal"] else "false"},\n'
            obj=dict[method]["obj"]
            if obj == 'N/A':
                obj='\"N/A\"'
            s+='\t\t"obj":'+f'{obj},\n'
            s+='\t\t"sol":'+f'{dict[method]["sol"]}\n'
            s+='\t},\n'
        s=s[:-2]+'\n'
        s+='}'
        return s

    def iterencode(self, o, _one_shot: bool = False) -> str:
        encoded_string=self.write_JSON_from_dict(o,_one_shot)

        return encoded_string
    
def saveJSON(content:dict,dzn_path,folder,format=False):
    os.makedirs(folder, exist_ok=True)
    filename = folder+'/'+os.path.splitext(os.path.basename(dzn_path))[0]+'.json'
    with open(filename, 'w') as file:
        indent= None if not format else 4
        json.dump(content, file, indent=indent)

def updateJSON(content:dict,dzn_path,folder,format=False,custom=True):
    # logic to preserve the order of the dict and remove ending '_'
    new_content = {}
    for key, value in content.items():
        if key.endswith('_'):
            new_content[key[:-1]] = value
        else:
            new_content[key] = value
    content.clear()
    content.update(new_content)
    os.makedirs(folder, exist_ok=True)
    filename = folder+'/'+os.path.splitext(os.path.basename(dzn_path))[0]+'.json'
    with open(filename, 'w') as file:
        indent= None if not format else 4
        if custom:
            json.dump(content, file, indent=indent, cls=CustomJSONEncoder)
        else:
            json.dump(content, file, indent=indent)


def saveModel(content,solver,inst_name,folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename=folder+f'{solver}_'+os.path.splitext(os.path.basename(inst_name))[0]+'.smt2'
    with open(filename,'w') as file:
        file.write(content)


# CHECKER
def run_checker(res_folder):
    check.main(['','./instances/', f'./{res_folder}/'])

# INSTANCES PARSING
def parse_dzn(dzn_path):
    data = {}
    with open(dzn_path, 'r') as file:
        content = file.read()
        content = content.replace('\n', '')
        values = content.split(';')
        
        for value in values:
            if '=' in value:
                key, value = value.split('=')
                key = key.strip()
                value = value.strip()
                
                if value.startswith('[') and value.endswith(']'):
                    if '|'.join(value).count('|') > 0:
                        # Handle 2D array
                        value = value[1:-1].strip().split('|')
                        array_2d = []
                        if len(value)>1:
                            for row in value:
                                if row:
                                    array_2d.append([int(v.strip()) for v in row.split(',')])
                            data[key] = array_2d
                        else:
                            # Handle 1D array
                            value = value[0].split(',')
                            data[key] = [int(v.strip()) for v in value]
                else:
                    data[key] = int(value)
    return data


def get_instance_number(instance_file):
    import re
    instance_number=re.search(r'inst(\d+)\.dzn', instance_file).group(1)
    return int(instance_number)