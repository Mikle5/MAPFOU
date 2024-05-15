import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from heapq import heappop, heappush, heapify
from random import randint, shuffle
import time
from sys import float_info
import warnings

from IPython.display import HTML
from PIL import Image, ImageDraw, ImageOps, ImageFont
from IPython.display import Image as Img
from IPython.display import display
from ipywidgets import IntProgress

from typing import Tuple, List, Iterable, Callable, Type, Dict, Union, Optional
import numpy.typing as npt


class Map:
    """
    Square grid map class represents the environment for our moving agent.

    Attributes
    ----------
    _width : int
        The number of columns in the grid

    _height : int
        The number of rows in the grid

    _cells : ndarray[int, ndim=2]
        The binary matrix, that represents the grid. 0 - cell is traversable, 1 - cell is blocked
    """

    def __init__(self, cells: npt.NDArray):
        """
        Initialization of map by 2d array of cells.

        Parameters
        ----------
        cells : ndarray[int, ndim=2]
            The binary matrix, that represents the grid. 0 - cell is traversable, 1 - cell is blocked.
        """
        self._width = cells.shape[1]
        self._height = cells.shape[0]
        self._cells = cells


    def in_bounds(self, i: int, j: int) -> bool:
        """
        Check if the cell (i, j) is on a grid.

        Parameters
        ----------
            i : int
                Number of the cell row in grid
            j : int
                Number of the cell column in grid
        Returns
        ----------
             bool
                Is the cell inside grid.
        """
        return (0 <= j < self._width) and (0 <= i < self._height)


    def traversable(self, i: int, j: int) -> bool:
        """
        Check if the cell (i, j) is not an obstacle.

        Parameters
        ----------
            i : int
                Number of the cell row in grid
            j : int
                Number of the cell column in grid
        Returns
        ----------
             bool
                Is the cell traversable.
        """
        if  self._cells[i, j] == 2:
            return 2
        if  self._cells[i, j] == 3:
            return 3
        if  self._cells[i, j] == 4:
            return 4
        return not self._cells[i, j]


    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Get a list of neighbouring cells as (i,j) tuples.
        It's assumed that grid is 4-connected (i.e. only moves into cardinal directions are allowed)

        Parameters
        ----------
            i : int
                Number of the cell row in grid
            j : int
                Number of the cell column in grid
        Returns
        ----------
            neighbors : List[Tuple[int, int]]
                List of neighbouring cells.
        """
        neighbors = []
        delta = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        for d in delta:
            if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1]):
                neighbors.append((i + d[0], j + d[1]))
        return neighbors


    def get_size(self) -> Tuple[int, int]:
        """
        Returns size of grid in cells.

        Returns
        ----------
            (height, widht) : Tuple[int, int]
                Number of rows and columns in grid
        """
        return (self._height, self._width)
    
    
    
class Node:
    """
    Node class represents a search node

    Attributes
    ----------
    i, j : int, int
        Coordinates of corresponding grid element.

    g : float | int
        g-value of the node (also equals time moment when the agent reaches the cell).

    h : float | int
        h-value of the node // always 0 for Dijkstra.

    f : float | int
        f-value of the node // always equal to g-value for Dijkstra.

    parent : Node
        Pointer to the parent-node.
    """

    def __init__(self,
                 i: int, j: int,
                 g: Union[float, int] = 0,
                 h: Union[float, int] = 0,
                 f: Union[float, int] = None,
                 parent: 'Node' = None):
        """
        Initialization of search node.

        Parameters
        ----------
        i, j : int, int
            Coordinates of corresponding grid element.
        g : float | int
            g-value of the node (also equals time moment when the agent reaches the cell).
        h : float | int
            h-value of the node // always 0 for Dijkstra.
        f : float | int
            f-value of the node // always equal to g-value for Dijkstra.
        parent : Node
            Pointer to the parent-node.
        """
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f
        self.parent = parent


    def __eq__(self, other):
        """
        Estimating where the two search nodes are the same,
        which is needed to detect dublicates in the search tree.
        """
        return self.i == other.i and self.j == other.j and self.g == other.g

    def __hash__(self):
        """
        To implement CLOSED as set/dict of nodes we need Node to be hashable.
        """
        return hash(str(self.i) + '-' + str(self.j))

    def __lt__(self, other):
        """
        Comparing the keys (i.e. the f-values) of two nodes,
        which is needed to sort/extract the best element from OPEN.
        """
        return self.f < other.f
    
    
class SearchTreePQD: # SearchTree which uses priority queue for OPEN and dict for CLOSED

    def __init__(self):
        self._open = []   # prioritized queue for the OPEN nodes
        self._closed = dict()         # dict for the expanded nodes = CLOSED
        self._enc_open_dublicates = 0  # the number of dublicates encountered in OPEN

    def __len__(self):
        """
        This gives the size of the search tree. Typically, we want to know
        the size of the search tree at the last iteration of the search
        to assess the memory footprint of the algorithm.
        """
        return len(self._open) + len(self._closed)

    def open_is_empty(self) -> bool:
        """
        open_is_empty should inform whether the OPEN is exhausted or not.
        In the former case the search main loop should be interrupted.
        """
        return len(self._open) == 0


    def add_to_open(self, item: Node):
        """
        Adding a (previously not expanded) node to the search-tree (i.e. to OPEN).
        It's either a totally new node (the one we never encountered before)
        or it can be a dublicate of the node that currently resides in OPEN.
        In this implementation we will detect dublicates lazily, thus at this
        point we dont care about them and just add a node to OPEN.
        """
        heappush(self._open, item)


    def get_best_node_from_open(self) -> Node:
        """
        Extracting the best node (i.e. the one with the minimal key) from OPEN.
        This node will be expanded further on in the main loop of the search.

        This is here where we must take care of the dublicates and discard the
        node if it was previously expanded (=resides in CLOSED) and take the next node.

        If OPEN becomes empty then we should return None.

        """
        while len(self._open) > 0:
            best_node = heappop(self._open)
            if not self.was_expanded(best_node):
                return best_node
            self._enc_open_dublicates += 1
        return None

    def add_to_closed(self, item: Node):
        self._closed[item] = item

    def was_expanded(self, item: Node) -> bool:
        return item in self._closed

    @property
    def opened(self):
        return self._open

    @property
    def expanded(self):
        return self._closed.values()

    @property
    def number_of_open_dublicates(self):
        return self._enc_open_dublicates
    
    

    
class CATable:
    """
    Class, which implements collision avoidance table for effective
    checking collisions with dynamic obstacles.

    Attributes
    ----------
    _pos_time_table : Dict[Tuple[int, int, int], int]
        Table that allows you to check cell (i, j) is occupied at time 0.
        If the cell is occupied, then the key (i, j, t) corresponds to a value
        equal to the ID of trajectory passing through the cell (i, j) at time t.

    _max_time_table : Dict[Tuple[int, int], int]
        A table that stores information about the time t starting from which the cell (i, j)
        will be permanently occupied. This is necessary to avoid a collision if
        the cell (i, j) is the final position of any trajectory, which means that
        starting from time t it is impossible to pass through it. If so,
        then the key (i, j) corresponds to the time t equal to the
        duration of trajectory that ends in this cell.

    _last_visit_table : Dict[Tuple[int, int], int | float]
        A table that stores information about time moment when the cell (i, j) was
        last occupied. This is necessary to verify that the agent remaining in its
        goal position (i, j) at time t1 will not collide with dynamic obstacle at
        time t2 > t1.
    """

    def __init__(self):
        self._pos_time_table = dict()
        self._max_time_table = dict()
        self._last_visit_table = dict()


    def add_trajectory(self, traj_id: int, trajectory: List[Tuple[int, int]]):
        """
        Adds trajectory to collision avoidance table. The first
        element of trajectory (trajectory[0]) corresponds to the
        position at time moment t = 0, secont element (trajectory[1])
        corresponds to the position at time moment t = 1 and etc.

        Parameters
        ----------
        traj_id : int
            Unique trajectory identifier.
        trajectory : List[Tuple[int, int]]
            A trajectory in the form of a sequence of (i, j).
        """

        for t, (i, j) in enumerate(trajectory):
            self._pos_time_table[(i, j, t)] = traj_id
            self._last_visit_table[(i, j)] = max(self._last_visit_table.get((i, j), -1), t)

        self._last_visit_table[trajectory[-1]] = math.inf
        self._max_time_table[trajectory[-1]] = len(trajectory) - 1


    def check_move(self, i1: int, j1: int, i2: int, j2: int, t_start: int) -> bool:
        """
        Check if the move between (i1, j1) and (i2, j2)
        at moment (t_start -> t_start+1) will not lead
        to the collision with other trajectories.

        Parameters
        ----------
        i1, j1 : int, int
            Position of start cell of move on the grid map.
        i2, j2 : int, int
            Position of target cell of move on the grid map.
        t_start : int
             Time when the move starts.

        Returns
        -------
        bool
            Is the move permissible (True) or will lead to the collision (False).
        """
        check_vertex = self.__check_pos_at_time(i2, j2, t_start + 1)
        check_edge = self.__check_rev_move(i1, j1, i2, j2, t_start)

        return check_vertex and check_edge


    def last_visited(self, i: int, j: int) -> int | float:
        """
        Returns value t when the cell (i, j) was last time occupied.
        If the cell was not occupied, then the value -1 should be returned.
        If, starting from some time moment, the cell is permanently occupied,
        then the value math.inf should be returned.

        Parameters
        ----------
        i, j : int, int
            Position of cell on the grid map.

        Returns
        -------
        int | float
            Value t when the cell (i, j) was last time occupied.

        """
        return self._last_visit_table.get((i, j), -1)


    def __check_pos_at_time(self, i: int, j: int, t: int) -> bool:
        """
        Checks, that cell (i, j) is occupied at momet t.

        Parameters
        ----------
        i, j: int, int
            Position of cell on the grid map.
        t : int
             Time moment to check.

        Returns
        -------
        bool
            False, if cell is occupied and True, if not.
        """
        return ((i, j, t) not in self._pos_time_table) and \
            (((i, j) not in self._max_time_table) or self._max_time_table[(i, j)] > t)


    def __check_rev_move(self, i1: int, j1: int, i2: int, j2: int, t_start: int) -> bool:
        """
        Checks, that there is not move along the same
        edge in one moment, but in reverse direction.

        Parameters
        ----------
        i1, j1 : int, int
            Position of start cell of move on the grid map.
        i2, j2 : int, int
            Position of target cell of move on the grid map.
        t_start : int
             Time when the move starts.

        Returns
        -------
        bool
            False, if there is move in reverse direction and True, if not.
        """
        if (i2, j2, t_start) in self._pos_time_table:
            id = self._pos_time_table[(i2, j2, t_start)]
            if (i1, j1, t_start + 1) in self._pos_time_table:
                return self._pos_time_table[(i1, j1, t_start + 1)] != id
        return True
    
    
    

def convert_string_to_cells(cell_str: str) -> npt.NDArray:
    """
    Converting a string (with '#' representing obstacles and '.' representing free cells) to a grid

    Parameters
    ----------
    cell_str : str
        String, which contains information about grid map ('#' representing obstacles
        and '.' representing free cells).

    Returns
    ----------
        cells : ndarray[np.int8, ndim=2]
            Grid map representation as matrix.
    """

    cells_list = []
    cells_row = []
    cell_lines = cell_str.split("\n")
    row = 0
    for line in cell_lines:
        cells_row = []
        column = 0
        if len(line) == 0:
            continue
        for char in line:
            if char == '.':
                cells_row.append(0)
            elif char == '#':
                cells_row.append(1)
            elif char == 'P':
                cells_row.append(2)
            elif char == 'O':
                cells_row.append(3)
            elif char == 'C':
                cells_row.append(4)
            else:
                continue
            column += 1
        row += 1
        cells_list.append(cells_row)
    cells = np.array(cells_list, dtype=np.int8)
    return cells

def compute_cost_timesteps(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    """
    Computes cost of simple action to move between cells (i1, j1) and (i2, j2) or wait.

    Parameters
    ----------
        i1 : int
            Number of the first cell row in grid.
        j1 : int
            Number of the first cell column in grid.
        i2 : int
            Number of the second cell row in grid.
        j2 : int
            Number of the second cell column in grid.

    Returns
    ----------
    int | float
        Cost of the action.
    """

    d = abs(i1 - i2) + abs(j1 - j2)
    if d == 0:  # wait
        return 1
    elif d == 1:  # cardinal move
        return 1
    else:
        raise Exception('Trying to compute the cost of non-supported move! ONLY cardinal moves are supported.')
      
    
def get_neighbors_timestep(
    i: int,
    j: int,
    t: int,
    grid_map: Map,
    ca_table: CATable
) -> List[Tuple[int, int]]:
    """
    Returns a list of neighbouring cells as (i, j) tuples.
    Function should return such neighbours, that allows only
    cardinal moves and also the current cell for case of wait action.

    Parameters
    ----------
    i, j: int, int
        Position of cell on the grid map.
    grid_map : Map
        Static grid map information.
    ca_table : CATable
        Collision avoidance table

    Returns
    -------
    neighbours : list[tuple[int, int]]
        List of neighbours (i, j) coordinates.
    """
    neighbours = []
    for neighbor in grid_map.get_neighbors(i, j) + [(i, j)]:
        if ca_table.check_move(i, j, *neighbor, t):
            neighbours.append(neighbor)
    return neighbours

def manhattan_distance(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    """
    Implementation of Manhattan heuristic.

    Parameters
    ----------
        i1, j1 : int, int
            (i, j) coordinates of the first cell row on grid.
        i2, j2 : int, int
            (i, j) coordinates of the second cell row on grid.

    Returns
    ----------
    int | float
        Manhattan distance between two cells.
    """
    return abs(i2 - i1) + abs(j2 - j1)

def astar_timesteps(
    task_map: Map,
    ca_table: CATable,
    start_i: int, start_j: int,
    goal_i: int, goal_j: int,
    steps_max: int,
    heuristic_func: Callable,
    search_tree: Type[SearchTreePQD]
) -> Tuple[
    bool,
    Optional[Node],
    int,
    int,
    Optional[Iterable[Node]],
    Optional[Iterable[Node]]
]:
    """
    Implementation of A* algorithm without re-expansion on dynamic obstacles domain.
    """

    ast = search_tree()
    steps = 0
    search_tree_size = 0

    start_node = Node(
        start_i, start_j,
        g=0, h=heuristic_func(start_i, start_j, goal_i, goal_j)
    )
    ast.add_to_open(start_node)

    while not ast.open_is_empty() and steps < steps_max:
        steps += 1
        best_node = ast.get_best_node_from_open()
        if best_node is None:
            break
        if best_node.i == goal_i and best_node.j == goal_j:
            return (True, best_node, steps, len(ast), ast.opened, ast.expanded)

        succ_coords = get_neighbors_timestep(
            best_node.i,
            best_node.j,
            best_node.g,
            task_map,
            ca_table
        )
        for coords in succ_coords:
            succ_node = Node(
                coords[0], coords[1],
                g=best_node.g + compute_cost_timesteps(
                    best_node.i, best_node.j, coords[0], coords[1]
                ),
                h=heuristic_func(coords[0], coords[1], goal_i, goal_j),
                parent=best_node,
            )
            if ast.was_expanded(succ_node):
                continue
            ast.add_to_open(succ_node)

        ast.add_to_closed(best_node)

    CLOSED = ast.expanded
    search_tree_size = len(ast)
    return False, None, steps, search_tree_size, None, CLOSED

def read_lists_from_file(path: str) -> List[List[Tuple[int, int]]]:
    """
    Auxiliary function that reads data from file in form of list of lists of pairs.

    Parameters
    ----------
    path : str
        Path to a file.
    Returns
    ----------
    List[List[Tuple[int, int]]]
        Data read from the file.
    """
    tasks_file = open(path)
    main_list = []
    curr_list = []
    for line_num, line in enumerate(tasks_file):
        if len(line) == 0:
            continue
        nums = tuple(map(int, line.split()))
        if len(nums) == 1:
            if line_num != 0:
                main_list.append(curr_list)
            curr_list = []
            continue
        curr_list.append(nums)
    main_list.append(curr_list)
    return main_list

def check_start_goal(
    start: Tuple[int, int], goal: Tuple[int, int],
    trajectory: List[Tuple[int, int]]
) -> bool:
    """
    Checks trajectory begins from start cell and ends in goal cell.
    """
    if trajectory[0] != start or trajectory[-1] != goal:
        return False
    return True


def process_trajectory(
    traj_id: int, trajectory: List[Tuple[int, int]],
    pos_time: Dict, max_times: Dict, last_times: Dict
):
    """
    Preparation of trajectory information in a form convenient for
    further checking for collisions with it.
    """
    for t, (i, j) in enumerate(trajectory):
        pos_time[(i, j, t)] = traj_id
        last_times[(i, j)] = max(last_times.get((i, j), -1), t)
    last_times[trajectory[-1]] = math.inf
    max_times[trajectory[-1]] = len(trajectory) - 1


def process_dyn_obstacles(
    dyn_obst_traj: List[List[Tuple[int, int]]]
) -> Tuple[Dict, Dict, Dict]:
    """
    Preparation of dynamic obstacles trajectories information in a form convenient for
    further checking for collisions with them.
    """
    pos_time = dict()
    max_times = dict()
    last_times = dict()

    for traj_id, trajectory in enumerate(dyn_obst_traj):
        process_trajectory(traj_id, trajectory, pos_time, max_times, last_times)
    return pos_time, max_times, last_times


def check_collisions(trajectory: List[Tuple[int, int]],
                     pos_time: Dict, max_times: Dict, last_times: Dict) -> bool:
    """
    Checks for collisions of the current trajectory with others.
    """
    for t1 in range(len(trajectory)-1):
        i1, j1 = trajectory[t1]
        t2 = t1+1
        i2, j2 = trajectory[t2]

        if (i2, j2, t2) in pos_time or \
            (((i2, j2) in max_times) and (max_times[(i2, j2)] <= t2)):
            return False
        if ((i1, j1, t2) in pos_time) and ((i2, j2, t1) in pos_time) and \
            (pos_time[(i1, j1, t2)] == pos_time[(i2, j2, t1)]):
            return False
    if len(trajectory) - 1 <= last_times.get(trajectory[-1], -1):
        return False
    return True

def draw_grid(draw_obj: ImageDraw, grid_map: Map, map_mask:Map, scale: Union[float, int]):
    """
    Draws static obstacles using draw_obj.
    """
    height, width = grid_map.get_size()
    for i in range(height):
        for j in range(width):
            # if grid_map.is_undefined_obstacle(i, j) and grid_map.traversable(i, j):
            #   draw_obj.rectangle((j * scale, i * scale, (j + 1) * scale - 1, \
            #                     (i + 1) * scale - 1), fill=(234, 237, 237), outline ="red", width=1)
            if not map_mask.traversable(i, j):
                outline_width = int(map_mask.is_undefined_obstacle(i, j))
                color = (234, 237, 237) if grid_map.traversable(i, j) else (70, 80, 80)
                draw_obj.rectangle((j * scale, i * scale, (j + 1) * scale - 1, \
                                (i + 1) * scale - 1), fill=color,
                                  outline ="red", width=outline_width)


def draw_start_goal(draw_obj: ImageDraw,
                    start: Tuple[int, int],
                    goal: Tuple[int, int],
                    scale: Union[float, int]):
    """
    Draws start and goal cells using draw_obj.
    """
    draw_obj.rounded_rectangle(
        ((start[1] + 0.1) * scale, (start[0] + 0.1) * scale,
        (start[1] + 0.9) * scale - 1, (start[0] + 0.9) * scale - 1),
        fill=(40, 180, 99), width=0.0, radius=scale * 0.22
    )
    draw_obj.rounded_rectangle(
        ((goal[1] + 0.1) * scale, (goal[0] + 0.1) * scale,
        (goal[1] + 0.9) * scale - 1, (goal[0] + 0.9) * scale - 1),
        fill=(231, 76, 60), width=0.0, radius=scale * 0.22
    )


def draw_dyn_object(
    draw_obj: ImageDraw,
    path: List[Tuple[int, int]],
    step: int,
    frame_num: int,
    frames_per_step: int,
    scale: Union[float, int],
    color: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    circle: bool
):
    """
    Draws position of dynamic object position at
    time t = (step + frame_num / frames_per_step)
    using draw_obj.
    """
    path_len = len(path)
    curr_i, curr_j = path[min(path_len - 1, step)]
    next_i, next_j = path[min(path_len - 1, step + min(frame_num, 1))]

    di = frame_num * (next_i - curr_i) / frames_per_step
    dj = frame_num * (next_j - curr_j) / frames_per_step

    corner_radius = scale * 0.3 if circle else scale * 0.22
    draw_obj.rounded_rectangle(
        (float(curr_j + dj + 0.2) * scale, float(curr_i + di + 0.2) * scale,
        float(curr_j + dj + 0.8) * scale - 1, float(curr_i + di + 0.8) * scale - 1),
        fill=color, width=round(0.03 * scale),
        outline=outline, radius=corner_radius
    )
def make_path(goal: Node) -> Tuple[List[Tuple[int, int]], Union[int, float]]:
    """
    Creates a path by tracing parent pointers from the goal node to the start node
    It also returns solution duration.

    Parameters
    ----------
    goal : Node
        Pointer to goal node in search tree.

    Returns
    ----------
    Tuple[List[Tuple[int, int]], float|int]
        Path and duration of the solution.
    """

    duration = goal.g
    current = goal
    path = []
    while current.parent:
        path.append((current.i, current.j))
        current = current.parent
    path.append((current.i, current.j))
    return path[::-1], duration
def prioritized_planning(
    task_map: Map,
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    max_steps: int,
    heuristic_func: Callable,
    choose_priority: Callable
) -> Tuple[
    bool,
    Optional[List[List[Tuple[int, int]]]],
    Optional[int],
    Optional[int],
    Optional[int]
]:
    """
    Implementation of Prioritized Planning algorithm.
    """
    idxs = tuple(choose_priority(starts, goals, heuristic_func))
    priorities = set()

    def iterate(priority):
        makespan = 0
        flowtime = 0
        steps = 0
        ca_table = CATable()
        paths = [[] for _ in priority]

        for j, id in enumerate(priority):
            start, goal = starts[id], goals[id]
            result = astar_timesteps(
                task_map, ca_table,
                *start, *goal,
                steps_max=max_steps,
                heuristic_func=heuristic_func,
                search_tree=SearchTreePQD
            )

            path, duration = make_path(result[1]) if result[0] else (None, None)
            if not result[0] or len(path) - 1 <= ca_table.last_visited(*path[-1]):
                if priority in priorities:
                    return False, paths, steps, makespan, flowtime
                priorities.add(priority)
                idxs = list(priority)
                idxs = tuple(idxs[j:j+1] + idxs[:j] + idxs[j+1:])
                return iterate(idxs)

            ca_table.add_trajectory(id, path)
            paths[id] = path
            steps += result[2]
            flowtime += duration
            makespan = max(makespan, duration)

        return True, paths, steps, makespan, flowtime

    return iterate(idxs)
def shortest_first(
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    heuristic_func: Callable
) -> List[int]:
    """
    Implementation of shortest-first priority construcion.
    Agents with a shorter distance (determined by heuristic function)
    from start to goal have a higher priority.

    Returns a sequence of agent IDs arranged according to their priority.
    It is assumed that the agent's ID correspond to its start/goal number
    in the corresponding lists.

    Parameters
    ----------
        starts : List[Tuple[int, int]]
            List of (i, j) coordinates of the initial positions of agents.
        goals : List[Tuple[int, int]]
            List of (i, j) coordinates of the target positions of agents.
        heuristic_func : Callable
            Implementation of heuristic function.

    Returns
    ----------
    List[int]
        A sequence of agent IDs arranged according to their priority.
        (higher priority = lower position in the list)
    """
    markers = [[*start, *goal] for start, goal in zip(starts, goals)]
    return sorted(
        range(len(starts)),
        key=lambda i: heuristic_func(*markers[i])
    )
def draw(grid_map: Map,
         map_mask: Map,
         starts: Optional[List[Tuple[int, int]]] = None,
         goals: Optional[List[Tuple[int, int]]] = None,
         paths: Optional[List[List[Tuple[int, int]]]] =  None,
         output_filename: str = 'animated_trajectories'):
    """
    Auxiliary function that visualizes the environment, agents paths and etc.

    Parameters
    ----------
    grid_map : Map
        Environment representation in for of grid.
    starts, goals : List[Tuple[int, int]] | None, List[Tuple[int, int]] | None
        Cells for start and goal positions of agents.
    paths : List[List[Tuple[int, int]]] | None
        List of sequences of cells, which represents the agents paths between
        start ang goal positions.
    output_filename : str
        Name of file with resulting animated visualization.
    """
    scale = 30
    quality = 6
    height, width = grid_map.get_size()
    h_im = height * scale
    w_im = width * scale
    dyn_obst_color = (204, 209, 209)
    dyn_obst_outline = (25, 25, 25)
    agent_colors = [(randint(10, 230), randint(10, 230), randint(10, 230)) for _ in starts]
    max_time = max([len(path) for path in paths]) if paths is not None else 1
    images = []

    for step in range(max_time):
        for n in range(0, quality):

            im = Image.new('RGB', (w_im, h_im), color = (234, 237, 237))
            draw = ImageDraw.Draw(im)
            draw_grid(draw, grid_map,map_mask, scale)

            if starts is not None and goals is not None:
                for a_id in range(len(starts)):
                    draw_start_goal(draw, starts[a_id], goals[a_id], scale)

            if paths is not None:
                for a_id, path in enumerate(paths):
                    draw_dyn_object(draw, path, step, n, quality, \
                                scale, agent_colors[a_id], agent_colors[a_id], True)

            im = ImageOps.expand(im, border=2, fill='black')
            images.append(im)

    images[0].save('./'+output_filename+'.png', save_all=True, \
                   append_images=images[1:], optimize=False, duration=500/quality, loop=0)
    display(Img(filename = './'+output_filename+'.png'))
    
def check_paths(starts: List[Tuple[int, int]],
                goals: List[Tuple[int, int]],
                paths: List[List[Tuple[int, int]]]) -> bool:
    """
    Checks that the presented solution is correct and
    there is not collision between agents trajectories.

    Parameters
    ----------
    starts, goals : List[Tuple[int, int]], List[Tuple[int, int]]
        Cells for start and goal positions of agent.
    paths : List[List[Tuple[int, int]]]
        List of sequences of cells, which represents the agents paths between
        start ang goal positions.

    Returns
    -------
    bool
        Is solution correct or not.
    """
    pos_time = dict()
    max_times = dict()
    last_times = dict()
    for agent_id, path in enumerate(paths):
        if not check_start_goal(starts[agent_id], goals[agent_id], path):
            return False
        if agent_id == 0:
            continue
        process_trajectory(agent_id-1, paths[agent_id-1], pos_time, max_times, last_times)
        if not check_collisions(path, pos_time, max_times, last_times):
            print(agent_id)
            print(path)
            return False
        prev_path = [path]
    return True
def simple_test(search_function: Callable, task: int, *args):
    """
    Function `simple_test` runs `search_function` on one task (use a number from 0 to 6 to
    choose a certain debug task on simple map or None to choose a random task from this pool)
    with *args as optional arguments and displays:
     - 'Path found!' and some statistics — path was found
     - 'Path not found!' — path was not found
     - 'Execution error' — an error occurred while executing the search_function
    In first two cases function also draws visualisation of the task

    Parameters
    ----------
    search_function : Callable
        Implementation of search method.
    task : int | None
        The number from 0 to 6 to choose a certain
        debug task on simple map or None to choose a
        random task from this pool.
    """

    height = 15
    width = 30
    map_str ='''
. . . . . . . . . .
. . P . . . . . . .
. # . . . . . . . .
. . . . P . . . . .
. . . . . . . . . .
. . O . . . . . . .
. . . . . . . . . .
. # . . . C . . . .
. . . . . . . . . .
. . . . . . . . . .
'''

    cells = convert_string_to_cells(map_str)
    task_map = Map(cells)


    all_starts = read_lists_from_file("data/mapf_st_starts.txt")
    all_goals = read_lists_from_file("data/mapf_st_goals.txt")

    if (task is None) or not (0 <= task < 7):
        task = randint(0, 6)

    starts = all_starts[task]
    goals = all_goals[task]
    try:
        result = search_function(task_map, starts, goals, *args)

        if result[0]:
            paths = result[1]
            steps = result[2]
            makespan = result[3]
            flowtime = result[4]
            correct = check_paths(starts, goals, paths)
            print("Path found! Steps: " + str(steps) + \
                    ". Makespan: " + str(makespan) + \
                    ". Flowtime: " + str(flowtime) + \
                    ". Correct: " + str(correct))
            draw(task_map, starts, goals, paths)
            print(paths)
        else:
            print("Path not found!")
    except Exception as e:
        print("Execution error")
        print(e)