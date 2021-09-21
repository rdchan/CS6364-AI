# to run, use `python cannibal_missionary_search.py | less`

here is an example output:
***** Uniform Cost Search  *****
At step 1
The frontier list is
[(1, 3, 0), (2, 3, 0), (2, 2, 0)]
The explored set is
{(3, 3, 1)}
------------
At step 2
The frontier list is
[(2, 2, 0), (2, 3, 0), (2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1)}
------------
At step 3
The frontier list is
[(2, 3, 0), (2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1), (2, 2, 0)}
------------
At step 4
The frontier list is
[(2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1), (2, 2, 0), (2, 3, 0)}
------------
At step 5
The frontier list is
[(0, 3, 0)]
The explored set is
{(2, 3, 1), (3, 3, 1), (1, 3, 0), (2, 2, 0), (2, 3, 0)}
------------
SOLUTION:
At state (3, 3, 1)
  Taking action (-2, 0, -1)

At state (1, 3, 0)
  Taking action (1, 0, 1)

At state (2, 3, 1)
  Taking action (-2, 0, -1)

At state (0, 3, 0)
  Taking action (1, 0, 1)

At state (1, 3, 1)
  Taking action (0, -2, -1)

At state (1, 1, 0)
  Taking action (1, 1, 1)

At state (2, 2, 1)
  Taking action (0, -2, -1)

At state (2, 0, 0)
  Taking action (1, 0, 1)

At state (3, 0, 1)
  Taking action (-2, 0, -1)

At state (1, 0, 0)
  Taking action (0, 1, 1)

At state (1, 1, 1)
  Taking action (-1, -1, -1)

At state (0, 0, 0)

***** Iterative Deepening Search  *****
At step 1
The successors to the current node, (3, 3, 1) are:
[(2, 3, 0), (1, 3, 0), (2, 2, 0)]
------------
At step 2
The successors to the current node, (3, 3, 1) are:
[(2, 3, 0), (1, 3, 0), (2, 2, 0)]
------------
At step 3
The successors to the current node, (2, 3, 0) are:
[(3, 3, 1)]
------------
At step 4
The successors to the current node, (1, 3, 0) are:
[(2, 3, 1), (3, 3, 1)]
------------
At step 5
The successors to the current node, (2, 2, 0) are:
[(2, 3, 1), (3, 3, 1)]
------------
SOLUTION:
At state (3, 3, 1)
  Taking action (-2, 0, -1)

At state (1, 3, 0)
  Taking action (1, 0, 1)

At state (2, 3, 1)
  Taking action (-2, 0, -1)

At state (0, 3, 0)
  Taking action (1, 0, 1)

At state (1, 3, 1)
  Taking action (0, -2, -1)

At state (1, 1, 0)
  Taking action (1, 1, 1)

At state (2, 2, 1)
  Taking action (0, -2, -1)

At state (2, 0, 0)
  Taking action (1, 0, 1)

At state (3, 0, 1)
  Taking action (-2, 0, -1)

At state (1, 0, 0)
  Taking action (0, 1, 1)

At state (1, 1, 1)
  Taking action (-1, -1, -1)

At state (0, 0, 0)

***** Greedy Best First Search  *****
At step 1
The frontier list is
[(1, 3, 0), (2, 3, 0), (2, 2, 0)]
The explored set is
{(3, 3, 1)}
------------
At step 2
The frontier list is
[(2, 2, 0), (2, 3, 0), (2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1)}
------------
At step 3
The frontier list is
[(2, 3, 0), (2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1), (2, 2, 0)}
------------
At step 4
The frontier list is
[(2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1), (2, 2, 0), (2, 3, 0)}
------------
At step 5
The frontier list is
[(0, 3, 0)]
The explored set is
{(2, 3, 1), (3, 3, 1), (1, 3, 0), (2, 2, 0), (2, 3, 0)}
------------
SOLUTION:
At state (3, 3, 1)
  Taking action (-2, 0, -1)

At state (1, 3, 0)
  Taking action (1, 0, 1)

At state (2, 3, 1)
  Taking action (-2, 0, -1)

At state (0, 3, 0)
  Taking action (1, 0, 1)

At state (1, 3, 1)
  Taking action (0, -2, -1)

At state (1, 1, 0)
  Taking action (1, 1, 1)

At state (2, 2, 1)
  Taking action (0, -2, -1)

At state (2, 0, 0)
  Taking action (1, 0, 1)

At state (3, 0, 1)
  Taking action (-2, 0, -1)

At state (1, 0, 0)
  Taking action (0, 1, 1)

At state (1, 1, 1)
  Taking action (-1, -1, -1)

At state (0, 0, 0)

***** A* Search  *****
At step 1
The frontier list is
[(1, 3, 0), (2, 3, 0), (2, 2, 0)]
The explored set is
{(3, 3, 1)}
------------
At step 2
The frontier list is
[(2, 2, 0), (2, 3, 0), (2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1)}
------------
At step 3
The frontier list is
[(2, 3, 0), (2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1), (2, 2, 0)}
------------
At step 4
The frontier list is
[(2, 3, 1)]
The explored set is
{(1, 3, 0), (3, 3, 1), (2, 2, 0), (2, 3, 0)}
------------
At step 5
The frontier list is
[(0, 3, 0)]
The explored set is
{(2, 3, 1), (3, 3, 1), (1, 3, 0), (2, 2, 0), (2, 3, 0)}
------------
SOLUTION:
At state (3, 3, 1)
  Taking action (-2, 0, -1)

At state (1, 3, 0)
  Taking action (1, 0, 1)

At state (2, 3, 1)
  Taking action (-2, 0, -1)

At state (0, 3, 0)
  Taking action (1, 0, 1)

At state (1, 3, 1)
  Taking action (0, -2, -1)

At state (1, 1, 0)
  Taking action (1, 1, 1)

At state (2, 2, 1)
  Taking action (0, -2, -1)

At state (2, 0, 0)
  Taking action (1, 0, 1)

At state (3, 0, 1)
  Taking action (-2, 0, -1)

At state (1, 0, 0)
  Taking action (0, 1, 1)

At state (1, 1, 1)
  Taking action (-1, -1, -1)

At state (0, 0, 0)

***** Recursive Best First Search  *****
At step 1
The successors to the current node, (3, 3, 1) are:
[(2, 3, 0), (1, 3, 0), (2, 2, 0)]
------------
At step 2
The successors to the current node, (2, 3, 0) are:
[(3, 3, 1)]
------------
At step 3
The successors to the current node, (1, 3, 0) are:
[(2, 3, 1), (3, 3, 1)]
------------
At step 4
The successors to the current node, (2, 2, 0) are:
[(2, 3, 1), (3, 3, 1)]
------------
At step 5
The successors to the current node, (2, 3, 1) are:
[(2, 2, 0), (1, 3, 0), (0, 3, 0)]
------------
SOLUTION:
At state (3, 3, 1)
  Taking action (-2, 0, -1)

At state (1, 3, 0)
  Taking action (1, 0, 1)

At state (2, 3, 1)
  Taking action (-2, 0, -1)

At state (0, 3, 0)
  Taking action (1, 0, 1)

At state (1, 3, 1)
  Taking action (0, -2, -1)

At state (1, 1, 0)
  Taking action (1, 1, 1)

At state (2, 2, 1)
  Taking action (0, -2, -1)

At state (2, 0, 0)
  Taking action (1, 0, 1)

At state (3, 0, 1)
  Taking action (-2, 0, -1)

At state (1, 0, 0)
  Taking action (0, 1, 1)

At state (1, 1, 1)
  Taking action (-1, -1, -1)

At state (0, 0, 0)
