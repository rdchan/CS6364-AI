Rishi Chandna

RDC180001

rdc180001@utdallas.edu

AI 6364.001 Homework 1

I used the Python language to implement the solutions for the programming assignments of this homework. There are three important files:
    - utils.py
    - cannibal_missionary_search.py
    - seattle_dallas_search.py
`utils.py` is from the tetxtbook and supports Priority Queues that are used in the other files.

`cannibal_missionary_search.py` and `seattle_dallas_search.py` include some classes and functions from the aima codebase file `search.py`. It uses the Graph class and supporting Node and Problem Solving agents. Additionally, I based my search algorithms off of the implementations in the aima code base. 

For the `cannibal_missionary_search.py`, I have written a `CannibalMissionary` class that follows the `Problem` abstraction from the aima code base. It is instantiated in the `main` function that gets called with a start state of (3, 3, 1) and a goal state of (0, 0, 0). To implement the actions() function, I have used a helper function called `is_valid_action()`. I iterate through all 10 possible actions, and find those that are valid given the state. This list is then returned in the actions() function. Since I have implemented the abstractions provided int he Problem class as described, I am able to pass the instantiation of the problem to the different search algorithms that are based off of the aima code base implementations. I have added additional code for generating easily readable output that satisfies the extra credit features of printing the first 5 steps in an easily comprehensible format. 

In the `seattle_dallas_search.py` file, I have defined an UndirectedGraph, using the aima code base for this class, and filled in the dictionary that initializes the graph with information from the road map provided in the instructions. To compute the heuristic (informed closeness of each node to the goal) I am using the straight line/flight distance from each city to Dallas inside of a lambda function. For the RBFS in problem 3.1, I based it off of the aima code base implementataion and included some additional logic that allow for clear output of the f_limit, best, alternative, current city and next city at each step of the algorithm. I've modified the best_first_graph_search in a similar manner so that it works as expected for problem 3.4 of the homework.

Additionally, I've written a function `BFS_heuristic_check` that iterates through all of the nodes in the US road map and checks if the heuristic is consistent starting at each of the nodes. It does so by using a breadth first search to explore all connected parts of the graph and checks that the child heuristic is never less than the heuristic of its parent. If this is true for any part of any path taken, then the `main()` function which calls the heuristic check will print that the heuristic is NOT consistent. This lines up with the definition of a consistent heuristic. As it turns out, using the straight line distance as provided is not a consistent heuristic, so using it for A* means that it is not proven to give the most efficient solution for all graph searches. Still, it will likely return the optimal solution for most searches since there was only 1 child/parent pair in which the heuristic was not consistent (with Bakersville). A simple fix would be to update the straight line distance from Bakersville to Dallas, as it seems to fail some sort of triangle inequality given the other meeasurements.

For the programming assignments in problem 2, run the Cannibal and Missionaries problem.

To run the Cannibal and Missionaries problem, do:
`python3 cannibal_missionary_search.py`
It will run all 5 searches in the following order:

    - Uniform Cost Search
    - Iterative Deepening Search
    - Greedy Best First Search
    - A* Search
    - Recursive Best First Search

The first five steps will be listed along with the frontier and explored/expanded sets. 
Following this initial print, the entire solution will be printed, all states and what actions were taken to transition between the states. 

----------------------

For problems 3.1 and 3.4, as well as the 3.4 extra credit, run the Seattle to Dallas search problem script.

To run the Seattle to Dallas search problem, do:
`python3 seattle_dallas_search.py`
It will run the Recursive Best First Search algorithm, followed by the A* search algorithm.
Consistent with the Extra Credit for part 4, it will check if the heuristic provided in the problem is consistent with the road graph. 
