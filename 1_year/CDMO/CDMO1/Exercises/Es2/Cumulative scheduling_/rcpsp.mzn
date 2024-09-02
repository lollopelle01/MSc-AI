include "globals.mzn";

int: n_tasks;   % Number of tasks
set of int: TASKS = 1..n_tasks;  
array[TASKS] of int: duration; % Task durations

int: n_res;   % Number of resources
set of int: RESOURCES = 1..n_res; 
array[RESOURCES] of int: l; % Resource capacities
array[RESOURCES,TASKS] of int: req; %Resource requirements of tasks

int: p;       % Number of precedence relations
set of int: PREC = 1..p;
array[PREC,1..2] of TASKS: pre; % Precedence relations

%decision and optimization variables and domains 
int: t = sum(duration);
array[TASKS] of var 0..t: start;

%precedence constraints
constraint forall(i in PREC)(
      start[pre[i,1]] + duration[pre[i,1]] <= start[pre[i,2]]
      );

%resource constraints
constraint forall(r in RESOURCES)(
    cumulative(start, duration, [req[r,t] | t in TASKS], l[r]));

/*
%implied constraint
constraint forall(r in RESOURCES, i,j in TASKS where i<j)
  (if req[r,i] + req[r,j] > l[r] then  disjunctive ([start[i], start[j]], [duration[i], duration[j]]) endif);
*/ 

%makespan as objective function
var int: makespan = max(t in TASKS)(start[t]+duration[t]);
solve minimize makespan;
%solve minimize max(t in TASKS)(start[t]+duration[t]);

output [
    "Start times = ", show(start), "\n",
    "End times   =", show([start[i]+duration[i] | i in TASKS]), "\n",
    "makespan = ", show(makespan), "\n"
];