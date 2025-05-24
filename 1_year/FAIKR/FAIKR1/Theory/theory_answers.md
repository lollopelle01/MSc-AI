# Theory answers for FAIKR1 exam

Collection of questions from past exams.
TODO : table of contents

## What are the main features of a swarm intelligence algorithm?

- These systems are usually composed of simple individuals with limited capacities.
- The individuals are not aware of the environment in which they live in.
- There exist local communication patterns between individuals. (es. stigmergy)
- These smart behaviour emerge autonomously with no central coordination nor supervision since there is distributed computation.
- Development of intelligent systems based on natural metaphores, that are robust and adaptive.

## What are the main approaches of deductive planning. Explain the main differences.

Dedictive planning uses logics for representing states, goals and actions and generates a plan as a theorem proof.
We have seen 2 approaches:

### 1) Green Formulation

- **Situation Calculus**: A formalism used to represent and reason about change in a logical framework.
  1. **Actions**: Events that cause changes in the world.
  2. **Situations**: Sequences of actions leading to a particular state of the world.
  3. **Fluents**: Properties that can change over time, dependent on the situation.
     This approach has high expressivity so it can describe complex problems.
- **Planning Based on Logic Resolution**:
  He finds a proof of a formula containing a state variable. At the end of the proof the state variable will be instantiated to the plan to reach the objective.

### 2) Kowalsky Formulation

- **Frame Problem**: Involves specifying what changes and what stays the same after an action occurs.
  - Aims to maintain relevance of states across actions without excessive repetition of unchanged information.
- **One Frame Assertion Per Action**:
  - Each action has an associated frame assertion.
  - The frame assertion explicitly states the effects of the action on the state variables.
  - This means for every action, there is a clear and concise assertion about how it changes the state of the world.
  - Simplifies the process of updating the state and helps manage the complexity of state transitions.

## What are metaheuristics? Describe the main algorithms that have been presented during the course.

TODO : verificare
A metaheuristic is a high-level procedure used in optimization problems used to find a sufficiently good solution for a problem.
We have seen many of them:

1. Stimulated Annealing
   ````
   TODO: algoritmo
   ````
2. Tabu Search
   ````
   TODO: algoritmo
   ````
3. Iterated local search
   ````
   TODO: algoritmo
   ````
4. Genetic algorithm
   ````
   TODO: algoritmo
   ````

## What is arc-consistency? Describe the algorithm to achieve it. Explain the properties of values that are removed from constraints and of values that are left in the domains.

Arc consistency is a property of constraint satisfaction problems (CSPs). A variable $X_i$ is arc-consistent with respect to another variable $X_j$ if and only if for every value $v_i$ in the domain of $X_i$, there is a value $v_j$ in the domain of $X_j$ such that $(v_i, v_j)$ satisfies the constraint between $X_i$ and $X_j$.
**All values removed by arc-consistency are not part of any feasible solution BUT values that are left in the domains are not necessarily part of a consistent solution.**

## What is conditional planning and which are its main features?

A conditional planner is a search algorithm that generates various alternative plans for each source of uncertainty of the plan.
It is constitued by:

- Causal actions, i.e. classic actions.
- Sensing actions, used to retrieve unknown information.
- Several alternative partial plans of which only one will be then executed. If one has n sensing actions and each sensing action has 2 possible outcomes, then one will have 2^n partial plans.
  Conditional planning, however, implies a combinatorial explosion of the search space and requires a lot of memory to be implemented. Moreover, not always all the alternative contexts are known in advance.

## What is modal truth criterion and why it has been defined.

When dealing with planners its important to provide some tools to be able to generate complete planners, meaning that whenever a solution exists, the planner will be able to get to it. MTC introduces some plan refinements methods, dealing with goal achievement and solving threats.

1. **Establishment**, open goal achievement by means of:
   - A new action to be inserted in the plan.
   - An ordering constraint with an action already in the plan.
   - A variable assignment, namely a uniﬁcation.
2. **Promotion**, which represents an ordering constraint that imposes the threatening action before the ﬁrst action of the causal link.
3. **Demotion**, which represents an ordering constraint that imposes the threatening action after the second action of the causal link.
4. **White knight**, which inserts a new operator, or one already in the plan, in order to re-establish the condition needed to execute the second action of the speciﬁc causal link. This is used when neither promotion nor demotion can be used, i.e. when the threatening action can only be put between the actions of the speciﬁc causal link.
5. **Separation**, which inserts non co-designation constraints between the variables of the threatening action’s eﬀect and the threatened action’s precondition so to avoid uniﬁcation. This is useful when variables have not yet been instantiated.
   Any threat imposed by stack (Y, c) can be solved by imposing X ≠ Y.

## What is Particle Swarm Optimization and which are its main features?

When dealing with Swarm algorithms, we could consider introducing some notion for the individuals, letting them able to create a sort of common objective, like searching food. Whenever an individual finds a food source, it can choose to get to the food source or stay within the group. If it occurs that more than a couple of individuals choose to get to the food, then there is a big chance that most of the population in the surrounding gets there too. So, each individual has to choose between an individualistic choice or a social one, weighting them on the basis of how many other individuals are doing each thing.
To optimize the problem, PSO sets up a population of candidate solutions: particles. These particles are then moved in the search space trough simple mathematical formulas. This motion is guided by the best position, in the search space, found so far, updated whenever new solutions are discovered. A solution is better than others if a cost function f calculated in that point has lower value than the previous solution, until no better solution exists.
After initializing each particle with a uniformly distributed random vector for its position, set as best position, and a velocity based on the position, compute the best-known position swarm-wise. Then for each particle generate random numbers and update velocity and position based on these numbers. Update the best-known positions if needed. Repeat this until certain termination criteria are reached, like a certain threshold of the cost function or a maximum number of iterations.

## What are non-informed search strategies? Describe the strategies that have been presented during the course.

Non-informed search strategies, also known as uninformed or blind search strategies, are algorithms used in artificial intelligence and computer science to explore problem spaces without using any information about the specific characteristics of the problem. These algorithms rely on systematic exploration of the search space rather than using domain-specific knowledge to guide the search. Here are some common non-informed search strategies:

- **Breadth-First Search** (BFS): BFS explores a tree or graph level by level, visiting all the nodes at the current depth before moving on to nodes at the next depth. It uses a queue data structure to keep track of the nodes to be expanded.
- **Depth-First Search** (DFS): DFS explores a tree or graph by going as deep as possible along one branch before backtracking. It uses a stack data structure to keep track of the nodes to be expanded.
- **Uniform-Cost Search** (UCS): UCS is a variant of BFS that considers the cost of the path to each node. It expands the node with the lowest path cost first. UCS is optimal in the sense that it finds the lowest-cost solution.
- **Depth-Limited Search** (DLS): DLS is a modification of DFS where a depth limit is imposed on the search. It limits the depth of exploration to avoid infinite loops in cases where the graph might be infinite.
- **Iterative Deepening Depth-First Search** (IDDFS): IDDFS is a combination of BFS and DFS. It performs DFS up to a certain depth limit and iteratively increases the depth limit until the goal is found.

Non-informed search strategies are generally less efficient than their informed (heuristic-based) counterparts in terms of finding solutions quickly, especially in large and complex problem spaces. However, they are useful in scenarios where little or no domain-specific information is available, or when the search space is relatively small and straightforward.

## What is hierarchical planning and explain the method presented during the course.

Hierarchical planners are search algorithms that manage the creation of complex plans at different levels of abstraction, by **considering the simplest details only after finding a solution for the most difficult ones**. Given a goal, the hierarchical planner performs a meta-level search to generate a meta-level plan which leads from a state that is very close to the initial one to a state which is very close to the goal. Next, the plan is the completed with a lower-level search which takes into account the omitted details.
Starting from the STRIPS algorithm, we specify a hierarchy of goals and preconditions assigning criticality values to each action for then proceed to the first level of abstraction: we consider only the preconditions with the higher criticality value, if the preconditions have lower criticality level they are fulfilled. In the next level of abstraction, we start again from the initial state, and we insert the actions computed on the previous level with their preconditions even if there are inconsistencies.

## What is Breadth-First Search? Describe this search strategy and discuss its completeness and complexity
It is a search strategy that always EXPANDS LESS DEEP tree nodes. It ensures COMPLETENESS, but we don’t have EFFICIENT IMPLEMENTATIONS.
In fact it has excessive memory footprint since it has an exponential complexity in space and time because in the worst case at a depth $d$ and with a branching factor $b$ we will have a maximum number of expanded nodes of $b^d$.

## What are the main features of a local search algorithm?
Local search algorithms start from an initial solution and iteratively try to imporve it through local moves. The algorithm defines a neighborhood around the current solution with that local moves, which is the set of solutions that can be reached from the current solution by making small, local changes. They incorporate mechanisms to escape local optima, such as simulated annealing or tabu search, to explore more of the search space.


## What are the main features of iterative deepening?
It is a search strategy that avoids the problem of choosing the maximum depth limit by trying all possible depth limits. It combines the depth-first and breadth first strategies. It is complete and explores one branch at a time. It has exponential complexity in time and linear in space. It is optimal if step cost is 1.