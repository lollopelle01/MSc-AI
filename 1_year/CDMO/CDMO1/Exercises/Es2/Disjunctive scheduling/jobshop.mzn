include "globals.mzn";

enum JOB;
enum TASK;
TASK: last = max(TASK);
array [JOB,TASK] of int: d;                   % task durations
int: total = sum(i in JOB, j in TASK)(d[i,j]);% total duration
int: digs = ceil(log(10.0,int2float(total))); % digits for output


array [JOB,TASK] of var 0..total: start;          % start times
var 0..total: makespan;                            % total end time 


constraint %% ensure the tasks occur in sequence
    forall(i in JOB) (
        forall(j in TASK where j <last) 
            (start[i,j] + d[i,j] <= start[i,enum_next(TASK,j)]) /\
        start[i,last] + d[i,last] <= makespan
    );


%% ensure no overlap of tasks 
%  disjunctive global constraint 
 constraint 
    forall(j in TASK) (
        disjunctive([start[i,j] | i in JOB], [d[i,j] | i in JOB])
    );      


% Decomposition of the disjunctive global constraint
/*

% my nooverlap predicate
predicate my_no_overlap(var int:s1, int:d1, var int:s2, int:d2) =
    s1 + d1 <= s2 \/ s2 + d2 <= s1;

constraint %% ensure no overlap of tasks
    forall(j in TASK)( 
        forall(i,k in JOB where i < k)(
            my_no_overlap(start[i,j], d[i,j], start[k,j], d[k,j])
          )
      );        
*/

solve :: int_search(start, input_order, indomain_min)  minimize makespan;


output ["makespan = \(makespan)\n"] ++
       [ show_int(digs,start[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in JOB, j in TASK ];
