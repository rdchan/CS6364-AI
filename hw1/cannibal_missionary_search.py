"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
import math
from collections import deque

from utils import *


print_frontier_steps = 0
class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________


class SimpleProblemSolvingAgentProgram:
    """
    [Figure 3.1]
    Abstract framework for a problem-solving agent.
    """

    def __init__(self, initial_state=None):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root)."""
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it."""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError


# ______________________________________________________________________________
# Uninformed Search algorithms

def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    print_frontier_steps = 0
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
        if (print_frontier_steps < 5):
            print("At step", print_frontier_steps + 1, "The current node is", node)
            print("The frontier list is")
            sorted_frontier = frontier.heap
            sorted_frontier.sort(key=lambda x: x[0])
            print([node[1].state for node in sorted_frontier])
            print("The explored set is")
            print(explored)
            print('------------')
            print_frontier_steps = print_frontier_steps + 1
    return None


def uniform_cost_search(problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, display)


explored_extra_credit = set()
frontier_extra_credit = set()

def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""
    def recursive_dls(node, problem, limit):
        global print_frontier_steps
        global explored_extra_credit
        global frontier_extra_credit
        explored_extra_credit.add(node)
        successors = node.expand(problem)
        for s in successors:
            frontier_extra_credit.add(s)
        for i in explored_extra_credit:
            if i in frontier_extra_credit:
                frontier_extra_credit.remove(i)
        if (print_frontier_steps < 5):
            print("At step", print_frontier_steps + 1,"The current node is", node)
            print("The frontier list is")
            print([x.state for x in frontier_extra_credit])
            print("The explored set is")
            print([x.state for x in explored_extra_credit])
            print("----------------")
            print_frontier_steps = print_frontier_steps + 1
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""

    global print_frontier_steps
    print_frontier_steps = 0
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ______________________________________________________________________________
# A* heuristics 

class CannibalMissionary(Problem):
    """ The problem of getting cannibals and missionaries to the other side of a river via a single boat, without letting the missionaries get eaten
    a state is represented as a tuple of length 3 (c, m, b) where
    c indicates the number of cannibals on the starting side
    m indicates the number of missionaries on the starting side
    b indicates if the boat is on the starting side (1) or on the ending side (0)
    For example, (3, 3, 1) is the start state, moving one missionary to the other side goes to state (3, 2, 0)
    The goal state is to have all the people on the finishing side, along with the boat, that is: (0, 0, 0)
    """
    def __init__(self, initial, goal=(0, 0, 0)):
        """ Define goal state and initialize a problem  """
        super().__init__(initial, goal)

    # make sure that the resulting state of following an action given a state is valid.
    def is_valid_action(self, state, action):
        resulting_state = self.result(state, action)
        # check if the boat moved in the right way
        if(resulting_state[2] != 0 and resulting_state[2] != 1):
            return False

        # check that there were no cannibals created or destroyed
        if(resulting_state[0] < 0 or resulting_state[0] > 3):
            return False

        # check that there were no missionaries created or destroyed
        if(resulting_state[1] < 0 or resulting_state[1] > 3):
            return False

        # check that the left side missionaries don't get eaten
        if(resulting_state[1] != 0 and resulting_state[0] > resulting_state[1]):
            return False

        # check that the right side missionaries don't get eaten
        if(resulting_state[1] != 3 and resulting_state[0] < resulting_state[1]):
            return False

        return True


    def actions(self, state):
        """ Return the actions that can be executed in the given state. The result would be a list since there are at most 5 possible actions
            Actions are represented as a tuple that gets added to the state. That is, to move 1 missionary and 1 cannibal from the starting to the finishing state, the action would be A = (-1, -1, -1)
            And then, to send a missionary back from the finishing state to the starting state, the action would be A = (0, 1, 1)
        """
        all_actions = [(0, -1, -1), (-1, 0, -1), (-2, 0, -1), (0, -2, -1), (-1, -1, -1), (0, 1, 1), (1, 0, 1), (2, 0, 1), (0, 2, 1), (1, 1, 1)]

        possible_actions = []

        # only include an action as possible if it is a valid action
        for action in all_actions:
            if self.is_valid_action(state, action):
                possible_actions.append(action)
        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        resulting_state = (state[0] + action[0], state[1] + action[1], state[2] + action[2])
        return resulting_state

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """
        return state == self.goal

# ______________________________________________________________________________
# Other search algorithms


def recursive_best_first_search(problem, h=None):
    """[Figure 3.26]"""
    h = memoize(h or problem.h, 'h')

    # keep track of each set across the recursive calls, as logically the algorithm shares these between each check
    global explored_extra_credit 
    global frontier_extra_credit
    global print_frontier_steps
    explored_extra_credit = set()
    frontier_extra_credit = set()
    print_frontier_steps = 0
    def RBFS(problem, node, flimit):
        global explored_extra_credit
        global frontier_extra_credit
        global print_frontier_steps
        explored_extra_credit.add(node)
        if problem.goal_test(node.state):
            return node, 0 # (The second value is immaterial)

        successors = node.expand(problem)
        for s in successors:
            frontier_extra_credit.add(s)
        for i in explored_extra_credit:
            if i in frontier_extra_credit:
                frontier_extra_credit.remove(i)
        if (print_frontier_steps < 5):
            print("At step", print_frontier_steps + 1,"The current node is", node)
            print("The frontier list is")
            print([x.state for x in frontier_extra_credit])
            print("The explored set is")
            print([x.state for x in explored_extra_credit])
            print("----------------")
            print_frontier_steps = print_frontier_steps + 1
        if (len(successors) == 0):
            return None, math.inf
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if (len(successors) > 1):
                alternative = successors[1].f
            else:
                alternative = math.inf
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, math.inf)
    return result



# ______________________________________________________________________________

# _____________________________________________________________________________
# The remainder of this file implements examples for the search algorithms.

# ______________________________________________________________________________
# Graphs and Graph Problems


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)

class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or math.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = math.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return math.inf


# Code to compare searchers on various problems.


# used to print the header for each of the 5 search algorithms
def banner(search_pattern):
    print("\n*****", search_pattern, " *****")
    print_frontier_steps = 0

# used to print the final solution given be each search algorithm
def pretty_print(goal_node):
    states = [node.state for node in goal_node.path()]
    states = ["At state " + str(state) for state in states]
    actions = goal_node.solution()
    actions = ["  Taking action " + str(action) + '\n' for action in actions]
    result = [None]*(len(states)+len(actions))
    result[::2] = states
    result[1::2] = actions
    print("SOLUTION:")
    for item in result:
        print(item)
def main():
    myboat = CannibalMissionary((3, 3, 1))
    banner("Uniform Cost Search")
    ucs = uniform_cost_search(myboat, display=False)
    pretty_print(ucs)
    banner("Iterative Deepening Search")
    ids = iterative_deepening_search(myboat)
    pretty_print(ids)
    banner("Greedy Best First Search")
    gbfs = greedy_best_first_graph_search(myboat, lambda node: node.depth, display=False)
    pretty_print(gbfs)
    banner("A* Search")
    astar = astar_search(myboat, lambda node: node.depth, display=False)
    pretty_print(astar)
    banner("Recursive Best First Search")
    rbfs = recursive_best_first_search(myboat, lambda node: node.depth)
    pretty_print(rbfs)

main()
