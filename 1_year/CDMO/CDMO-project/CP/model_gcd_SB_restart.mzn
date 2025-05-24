include "globals.mzn";

% number of couriers
int: num_couriers;
set of int: COURIERS = 1..num_couriers;
% number of items
int: num_items;
set of int: ITEMS = 1..num_items;
% number of nodes(including the depot)
set of int: NODES = 1..num_items + 1;
int: depot=num_items + 1;

% couriers load capacity 
array[COURIERS] of int: courier_capacities;
% items size 
array[ITEMS] of int: item_sizes;
% matrix of distances between nodes
set of int:DELIVERS = 1..num_items-num_couriers+1;
array[NODES, NODES] of int: distances;

%%%%%%%%%%%%%%% DECISION VARIABLES %%%%%%%%%%%%%%%

% route taken by each courier, represented as a sequence of nodes
array[COURIERS, DELIVERS] of var NODES: stops;

% who is responsible for the delivery of a specific item
array[ITEMS] of var COURIERS: item_responsibility;

% bounds for the objective function
int:lower_bound;
int:upper_bound;

% total distance traveled by each courier
array[COURIERS] of var 0..upper_bound: distances_traveled;
% largest distance among all couriers
var lower_bound..upper_bound: longest_trip;

%%%%%%% CONSTRAINT FOR DECISION VARIABLES %%%%%%%%

% capacity constraint: the sum of item sizes assigned to each courier must not exceed the courier's capacity
constraint bin_packing_capa(courier_capacities, item_responsibility, item_sizes);

% each courier must be responsible for delivering at least one item
constraint nvalue(num_couriers,item_responsibility);

% each courier have to stop in at least one location different from the depot
constraint forall(c in COURIERS)(
  stops[c, 1] != depot
);

% all items must be delivered(in theory it's already guaranteed by 'bin_packing_capa' but helps with the performances)
constraint forall(i in ITEMS)(
  count(stops, i, 1)
);

% channeling constraint: the number of items delivered by a courier in 'item_responsibility' matches the stops in 'stops'
constraint forall(c in COURIERS)(
  forall(i in DELIVERS)(
    i <= count(item_responsibility, c) <-> stops[c, i] != depot
));

% all the items assigned to a courier in 'item_responsibility' must appear in the courier's route in 'stops'
constraint forall(c in COURIERS, i in ITEMS)(
  if item_responsibility[i] == c then
    member(row(stops, c), i) endif
);

% compute total distances for each courier
% compute the successor nodes for each item to optimize distance calculation (accessing directly to distances[stops[c, i],  stops[c, i+1]] it's extremely slower)
array[ITEMS] of var NODES:successors; 
constraint forall(c in COURIERS)(
  forall(i in 1..num_items-num_couriers)(
    if stops[c, i] != depot then
      successors[stops[c,i]] = stops[c, i+1] endif
  )
);

% the distance traveled by each courier is the sum of the distances between a node and its successor 
% + the distance from the depot to the first item delivered
% + the distance from the last item delivered to the depot
% (in this way we guarantee that the courier's route starts and ends at the depot)
constraint forall(c in COURIERS)(
    distances_traveled[c] = sum([distances[i, successors[i]]| i in ITEMS where item_responsibility[i] = c]) 
    + distances[depot, stops[c, 1]]
    + distances[stops[c, num_items-num_couriers+1], depot]
);

%%%%%%%%%% SIMMETRY BREAKING CONTRAINT %%%%%%%%%%%

constraint forall(c1 in 1..num_couriers-1)(
    forall(c2 in c1+1..num_couriers)(
      if courier_capacities[c1]=courier_capacities[c2] then
          stops[c1,1] < stops[c2,1] endif
    )
);

% minimize the longest trip distance among all couriers
constraint longest_trip = max(distances_traveled);
solve 
:: restart_luby(200) 
:: relax_and_reconstruct(array1d(stops), 90) 
:: seq_search([
  int_search(stops, first_fail, indomain_random),
  int_search(item_responsibility, first_fail, indomain_random),
%   int_search([longest_trip], first_fail, indomain_min)
])
minimize(longest_trip);

%%%%%%%%%%%%%%%% OUTPUT UTILITIES %%%%%%%%%%%%%%%%
% output ["Stops:\n"] ++
%        [ show(stops[i,j])++ 
%          if j == num_items then "\n" else ", " endif |
%          i in COURIERS, j in ITEMS ]++["\n"];

% output ["Itinerary:\n"] ++
%        [ 
%         % if j==1 then "Start, " else "" endif ++
%         if fix(stops[i,j])==depot then "" else show(stops[i,j])++", " endif ++ 
%         if j == num_items then 
%           % "End"++
%           " --> Distance="++show(distances_traveled[i])++"\n" 
%         else "" endif
%         |i in COURIERS, j in ITEMS
%        ];
       
% output["Obj: "++show(longest_trip)++"\n"];