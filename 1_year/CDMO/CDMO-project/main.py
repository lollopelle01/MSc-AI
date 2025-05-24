import os
from utils.utils import *
from datetime import timedelta
from argparse import ArgumentParser,RawTextHelpFormatter
this_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_dir)

class Configuration:
    def __init__(self) -> None:
        self.instances_folder='instances_dzn'
        self.instances_names=self.read_instances()
        self.smt_models_folder='SMT/models'
        self.results_folder='res'
        self.indent_results=True
        self.first=1
        self.last=21
        self.timeout=300
        self.run_checker=True

    def read_instances(self):
        return [self.instances_folder+'/'+instance for instance in sorted(os.listdir(self.instances_folder)) if instance.endswith('.dzn')]

def load_config(config_path, configuration:Configuration):
    from check_solution import read_json_file
    config = read_json_file(config_path)
    try:
        configuration.instances_folder=config['instances_folder']
        configuration.instances_names=configuration.read_instances()
        configuration.smt_models_folder=config['smt_models_folder']
        configuration.results_folder=config['results_folder']
        configuration.indent_results=config['indent_results']
        configuration.first=config['first']
        configuration.last=config['last']
        configuration.timeout=config['timeout']
        configuration.run_checker=config['run_checker']
    except KeyError:
        raise KeyError('Invalid configuration file. It must contain:\n- instances_folder\n- smt_models_folder\n- results_folder\n- indent_results\n- first\n- last\n- timeout\n- run_checker')


def main(args):
    configuration=Configuration()
    RUN_CP=RUN_SAT=RUN_SMT=RUN_MIP=True
    approach=args.cp or args.sat or args.smt or args.mip
    if (args.instances or approach) and not args.config:
        configuration.results_folder+='_temp'
        if os.path.exists(configuration.results_folder) and not args.keep_folder:
            import shutil
            shutil.rmtree(configuration.results_folder)
    if args.config:
        load_config(args.config,configuration)
    instances_numbers=[]
    if args.instances:
        for i in args.instances:
            if i<1 or i>21:raise ValueError(f'Invalid instance number. It must be in [1,21], instead got {i}')
        instances_numbers=args.instances
    else: 
        instances_numbers=[*range(configuration.first,configuration.last+1)]
    if approach:
        RUN_CP=RUN_SAT=RUN_SMT=RUN_MIP=False
    if args.cp:  RUN_CP =True
    if args.sat: RUN_SAT=True
    if args.smt: RUN_SMT=True
    if args.mip: RUN_MIP=True

    executable_instances=sorted([configuration.instances_names[i-1] for i in instances_numbers])

    # ============
    # |    CP    |
    # ============
    if RUN_CP:
        import CP.CP_launcher as CP
        CP_models = {
            'model_chfd.mzn': {
                'solvers': {
                    'chuffed': [
                        ('', {'timeout': timedelta(seconds=configuration.timeout), 'free_search': False}),
                    ]
                }
            },
            'model_chfd_SB.mzn': {
                'solvers': {
                    'chuffed': [
                        ('', {'timeout': timedelta(seconds=configuration.timeout), 'free_search': False}),
                    ]
                }
            },
            'model_gcd.mzn': {
                'solvers': {
                    'gecode': [
                        ('', {'timeout': timedelta(seconds=configuration.timeout)}),
                    ],
                }
            },
            'model_gcd_SB.mzn': {
                'solvers': {
                    'gecode': [
                        ('', {'timeout': timedelta(seconds=configuration.timeout)}),
                    ],
                }
            },
            'model_gcd_restart.mzn': {
                'solvers': {
                    'gecode': [
                        ('', {'timeout': timedelta(seconds=configuration.timeout)}),
                    ]
                }
            },
            'model_gcd_SB_restart.mzn': {
                'solvers': {
                    'gecode': [
                        ('', {'timeout': timedelta(seconds=configuration.timeout)}),
                    ]
                }
            },
        }

        print('Solving with CP:')
        for instance_file in executable_instances:
            print(f'  Solving {instance_file}...')
            instance_results={}
            for model in CP_models:
                if ('restart' in model 
                    and get_instance_number(instance_file) < 11): continue
                for solver in CP_models[model]['solvers']:
                    for param_name,params in CP_models[model]['solvers'][solver]:
                        print(f'\tUsing {model} with {solver}-{param_name}...')
                        key=f'{os.path.splitext(model)[0]}_{solver}_{param_name}'
                        instance_results[key]=CP.solve_instance(instance_file,
                                                                model,
                                                                solver,
                                                                params)
                        updateJSON(instance_results,instance_file,configuration.results_folder+'/CP/',format=configuration.indent_results)

    # ============
    # |    SAT   |
    # ============
    if RUN_SAT:
        import SAT.SAT_launcher as SAT

        SAT_models = {
            'circuit':[
                ('ILU', {
                    'params_name':'ILU_SB',
                    'params':{'timeout': timedelta(seconds=configuration.timeout)},
                    'model_params':{
                        'incremental_factor' : 30, 
                        'symmetry' : True
                    }
                }),
                ('ILU', {
                    'params_name':'ILU',
                    'params':{'timeout': timedelta(seconds=configuration.timeout)},
                    'model_params':{
                        'incremental_factor' : 30, 
                        'symmetry' : False
                    }
                }),
                ('BS', {
                    'params_name':'BS_SB',
                    'params':{'timeout': timedelta(seconds=configuration.timeout)},
                    'model_params':{
                        'binary_cut' : 2,
                        'symmetry' : True
                    }
                }),
                ('BS', {
                    'params_name':'BS',
                    'params':{'timeout': timedelta(seconds=configuration.timeout)},
                    'model_params':{
                        'binary_cut' : 2,
                        'symmetry' : False
                    }
                }),
            ],
            '1-hot-cube':[
                ('BS', {
                    'params_name':'BS2_SB',
                    'params':{'timeout': timedelta(seconds=configuration.timeout)},
                    'model_params':{
                        'binary_cut' : 2,
                        'symmetry' : True
                    }
                }),
                ('BS', {
                    'params_name':'BS2',
                    'params':{'timeout': timedelta(seconds=configuration.timeout)},
                    'model_params':{
                        'binary_cut' : 2,
                        'symmetry' : False
                    }
                }),
            ],
        }
        
        print('Solving with SAT:')
        for instance_file in executable_instances:
            print(f'  Solving {instance_file}...')
            instance_results={}
            for model in SAT_models :
                for search_name, params in SAT_models[model] :
                    print(f'\tUsing {model}-{params["params_name"]}...')
                    result=SAT.solve_instance(instance_file,
                                              model,
                                              search_name,
                                              params['params'],
                                              params['model_params'],
                                              verbose_search=False, 
                                              verbose_solver=False
                                            )
                    instance_results[f'{model}_{params["params_name"]}'] = result
                    updateJSON(instance_results,instance_file,configuration.results_folder+'/SAT/',format=configuration.indent_results)

    # ============
    # |    SMT   |
    # ============
    if RUN_SMT:
        import SMT.SMT_launcher as SMT

        SMT_models = {
                    'z3': [
                        ('OPT', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'z3':True
                                }
                            }),
                        ('arrays_SB', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'simmetry_method':'>',
                                'use_arrays':True,
                                }
                            }),
                        ('arrays', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'simmetry_method':'None',
                                'use_arrays':True,
                                }
                            }),
                        ('best', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'simmetry_method':'>',
                                'best':True
                                }
                            }),
                    ],
                    'cvc5': [
                        ('arrays_SB', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'simmetry_method':'>',
                                'use_arrays':True,
                                }
                            }),
                        ('arrays', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'simmetry_method':'None',
                                'use_arrays':True,
                                }
                            }),
                        ('best', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':{
                                'simmetry_method':'>',
                                'best':True
                                }
                            }),
                    ],
        }
        
        print('Solving with SMT:')
        for instance_file in executable_instances:
            print(f'  Solving {instance_file}...')
            instance_results={}
            for solver in SMT_models:
                for param_name,params in SMT_models[solver].copy():
                    print(f'\tUsing {solver}-{param_name}...')
                    result,model=SMT.solve_instance(instance_file,
                                        solver,
                                        params['params'],
                                        params['model_params'],
                                        verbose=False)
                    instance_results[f'{solver}_{param_name}'] = result
                    if model:saveModel(model,solver,instance_file,f'SMT/models/{solver}/{param_name}/')
                    updateJSON(instance_results,instance_file,configuration.results_folder+'/SMT/',format=configuration.indent_results)
    
    # ============
    # |    MIP   |
    # ============    
    if RUN_MIP:
        import MIP.MIP_launcher as MIP
        MIP_models = {
                'solvers': {
                    'cbc': [
                        ('enum_all', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':None
                            }),
                        ('MTZ', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':None
                            }),
                    ],
                    'glpk': [
                        ('enum_all', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':None
                            }),
                        ('MTZ', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':None
                            }),
                    ],
                    'HiGHS': [
                        ('enum_all', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':None
                            }),
                        ('MTZ', {
                            'params':{'timeout': timedelta(seconds=configuration.timeout)},
                            'model_params':None
                            }),
                    ]
                    
                }
            }
        print('Solving with MIP:')
        for instance_file in executable_instances:
            print(f'  Solving {instance_file}...')
            instance_results={}
            for solver in MIP_models['solvers']:
                for model_name, params in MIP_models['solvers'][solver].copy():
                    if not (int(instance_file[-6:-4])>10 and 'enum_all' in model_name):      
                        print(f'\tUsing {solver}-{model_name}...')              
                        result,model=MIP.solve_instance(instance_file,
                                            solver,
                                            model_name,
                                            params['params'],
                                            params['model_params'],
                                            verbose=False)
                        instance_results[f'{solver}_{model_name}'] = result
                        updateJSON(instance_results,instance_file,configuration.results_folder+'/MIP/',format=configuration.indent_results)

    if configuration.run_checker:run_checker(configuration.results_folder)

if __name__=='__main__':
    desc='''When run with no arguments all the instances_numbers will be solved with all the approaches. The user can choose if solving one specific instance and/or approach.
The user can also specify a config file to change settings such as folder names or instances_numbers to execute (in the form [first,last], both included).

The JSON *must* be like this:
{
    "instances_folder": "instances_dzn",
    "smt_models_folder": "SMT/models",
    "results_folder": "res",
    "indent_results": true,
    "first": 1,
    "last": 21,
    "timeout": 300,
    "run_checker": true
}

If an instance or an approach is parsed, the results will be stored in a '<results_folder>_temp' folder. It will be cleared with the next execution if the -keep-folder flag is not set to True
If a directory is specified in the configuration JSON, the folder won't be cleared, but the instances results will be overwritten.'''
    parser=ArgumentParser(description=desc,formatter_class=RawTextHelpFormatter,allow_abbrev=True)
    parser.add_argument('-instances',type=int, nargs='+',required=False,help='Instance number to execute.')
    parser.add_argument('-config',type=str,required=False,help='JSON configuration file.')
    parser.add_argument('-cp',action='store_true',help='Run cp.')
    parser.add_argument('-sat',action='store_true',help='Run sat.')
    parser.add_argument('-smt',action='store_true',help='Run smt.')
    parser.add_argument('-mip',action='store_true',help='Run mip.')
    parser.add_argument('-keep_folder',action='store_true',help='Do not empty the folder when executing again.')
    main(parser.parse_args())