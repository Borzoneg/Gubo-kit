import py_trees as pt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from numpy import ndarray
from ros import SendStrClient

class BtViewer():
    def __init__(self, tree:pt.trees.BehaviourTree, blackboard: pt.blackboard.Client = None, figsize: tuple[float, float] = (18,3),
                 width_node: float = 0.6, height_node: float = 0.4, padding_node: list[float, float] = [0.1, 0.2]):

        self.tree = tree        
        self.blackboard = blackboard
        plt.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.manager.set_window_title(tree.root.name)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.axis('off')
        
        self.p_n_x = padding_node[0]
        self.p_n_y = padding_node[1]
        self.w_n = width_node
        self.h_n = height_node

        self.status_color = {'failure': 'red', 'running': 'yellow', 'success': 'green'}
        
        self.min_x, self.max_x, self.min_y, self.max_y = float('inf'), -float('inf'), float('inf'), -float('inf')
        
        self.node_status_patch = {}

        self._draw_node(self.tree.root, (0.0,0.0))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
                
        self.ax.set_xbound(self.min_x - (self.w_n + self.p_n_x)/2, self.max_x + (self.w_n + self.p_n_x)/2)
        self.ax.set_ybound(self.min_y - (self.h_n + self.p_n_y)/2, self.max_y + (self.h_n + self.p_n_y)/2)
        self.ax.set_aspect('equal', adjustable='box')
    
    def _draw_node(self, node, xy: tuple[float, float], linewidth: int = 2, fontsize=9):
        """recursive function to draw a node and all his children, calling this on the root will draw the entire tree
        Args:
            node (_type_): _description_
            xy (tuple[float, float]): _description_
            linewidth (int, optional): _description_. Defaults to 2.
            fontsize (int, optional): _description_. Defaults to 9.
        """
        
        rect = patches.Rectangle(np.array(xy) - np.array([self.w_n/2 + self.p_n_x / 4, self.h_n/2 + self.p_n_y / 4,]), self.w_n + self.p_n_x / 2, self.h_n + self.p_n_y / 2, zorder=1, visible=False)
        self.ax.add_patch(rect)
        self.node_status_patch[node.name] = rect

        self.min_x = xy[0] if xy[0] < self.min_x else self.min_x
        self.max_x = xy[0] if xy[0] > self.max_x else self.max_x
        self.min_y = xy[1] if xy[1] < self.min_y else self.min_y
        self.max_y = xy[1] if xy[1] > self.max_y else self.max_y
        if isinstance(node, pt.composites.Composite):
            children_y = xy[1] - (self.h_n + self.p_n_y)
            if (len(node.children) % 2) == 0:
                start_x_child = xy[0] - ((self.w_n + self.p_n_x)/2 + ((len(node.children)-2)/2 * (self.w_n + self.p_n_x)))
            else:
                start_x_child = xy[0] - (len(node.children)//2 * (self.w_n + self.p_n_x))
            
            # draw a line from this node going down, it will connect with the children
            self.ax.plot((xy[0], xy[0]), (xy[1] - self.h_n/2, xy[1] - self.h_n/2 - self.p_n_y/2), color='black', zorder=2)
            for i, child in enumerate(node.children):
                children_x = start_x_child + i * (self.w_n + self.p_n_x)
                if i != 0 and isinstance(child, pt.composites.Composite) and isinstance(node.children[i-1], pt.composites.Composite):
                    # if two composites are close we need to space them accoridngly with the number of children they have
                    # we do it on start child so that when we move one all of them move accoridngly
                    children_x += ((len((node.children[i-1]).children) / 2) * (self.w_n + self.p_n_x)) + ((len(child.children) - 2) / 2 * (self.w_n + self.p_n_x))
                    start_x_child += ((len((node.children[i-1]).children) / 2) * (self.w_n + self.p_n_x)) + ((len(child.children) - 2) / 2 * (self.w_n + self.p_n_x))
                if i == 0 or i == len(node.children)-1: # draw a horizontal line that connect parent with all children 
                    self.ax.plot((xy[0], children_x), (xy[1] - self.h_n/2 - self.p_n_y/2, xy[1] - self.h_n/2 - self.p_n_y/2), color='black', zorder=2)
                # draw a line from child going up, it will connect with parent node
                self.ax.plot((children_x, children_x),  (children_y + self.h_n/2, children_y + self.h_n/2 + self.p_n_y/2), color='black', zorder=2)
                self._draw_node(node=child, xy=(children_x, children_y), linewidth=linewidth, fontsize=fontsize)
            self._draw_composite(node=node, xy=xy, linewidth=linewidth, fontsize=fontsize)

        elif isinstance(node, pt.behaviour.Behaviour):
            self._draw_behaviour(behaviour=node, xy=xy, linewidth=linewidth, fontsize=fontsize)

    def _draw_composite(self, node: pt.composites.Composite, xy: tuple[float, float], fontsize: int, linewidth: int = 2):
        if isinstance(node, pt.composites.Sequence):
            rect = patches.Rectangle(np.array(xy) - np.array([self.w_n/2, self.h_n/2]), self.w_n, self.h_n, linewidth=linewidth, edgecolor='black', facecolor='orange', zorder=2)
            self.ax.add_patch(rect)
            self.ax.annotate(node.name.replace("_", "\n"), np.array(xy), color='black', fontsize=fontsize, ha='center', va='center')
        
        if isinstance(node, pt.composites.Selector):
            vertices = self._generate_octagon_verts(xy, width=self.w_n, height=self.h_n)
            hex = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='cyan', linewidth=linewidth, zorder=2)
            self.ax.add_patch(hex)
            self.ax.annotate(node.name.replace("_", "\n"), (vertices.mean(axis=0)), color='black', fontsize=fontsize, ha='center', va='center')
        
        if isinstance(node, pt.composites.Parallel):
            vertices = self._generate_parallelogram_verts(xy, width=self.w_n, height=self.h_n)
            parallelogram = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='yellow', linewidth=linewidth, zorder=2)
            self.ax.add_patch(parallelogram)
            self.ax.annotate(node.name.replace("_", "\n"), (vertices.mean(axis=0)), color='black', fontsize=fontsize, ha='center', va='center')

    def _draw_behaviour(self, behaviour: pt.behaviour.Behaviour, xy: tuple[float, float], fontsize: int, linewidth: int = 2):
        rect = patches.Rectangle(np.array(xy) - np.array([self.w_n/2, self.h_n/2]), self.w_n, self.h_n, linewidth=linewidth, edgecolor='black', facecolor='gray', zorder=2)
        self.ax.add_patch(rect)
        self.ax.annotate(behaviour.name.replace("_", "\n"), np.array(xy), color='black', fontsize=fontsize, ha='center', va='center')

    def _draw_status(self, node):
        if isinstance(node, pt.composites.Composite):
            for i, child in enumerate(node.children):
                self._draw_status(node=child)
        if node.status.name.lower() != 'invalid':
            color = self.status_color[node.status.name.lower()]
            self.node_status_patch[node.name].set_visible(True)
            self.node_status_patch[node.name].set_color(color)
        else:
            self.node_status_patch[node.name].set_visible(False)

    def _generate_hexagon_verts(self, centre: tuple[float, float]) -> ndarray:
        # width set the radius of the circle in which we inscribe the hexagon
        # verts = np.array(centre) + np.array([[ np.cos(np.pi), 0.0], [-np.cos(np.pi/4), np.sin(np.pi/4)], [ np.cos(np.pi/4), np.sin(np.pi/4)], [ np.cos(0), 0.0], [ np.cos(np.pi/4), -np.sin(np.pi/4)], [-np.cos(np.pi/4), -np.sin(np.pi/4)], ]) * width
        # width and height set the 4 points that would make a rectangle and then we find the two laying on x=0
        # verts = np.array(centre) + np.array([[-1/2, 1/2], [ 1/2, 1/2], [ 1, 0], [ 1/2, -1/2], [-1/2, -1/2], [ -1, 0], ]) * np.array([width, height])
        # width and height sets the rectangle we inscribe the hexagon in
        if self.w_n < self.h_n:
            print("Hexagon generation impossible")
            return None
        verts =  np.array(centre) + np.array([
                                              [-self.w_n/2+self.h_n/2,  self.h_n/2], 
                                              [ self.w_n/2-self.h_n/2,  self.h_n/2], 
                                              [ self.w_n/2,    0], 
                                              [ self.w_n/2-self.h_n/2, -self.h_n/2], 
                                              [-self.w_n/2+self.h_n/2, -self.h_n/2], 
                                              [-self.w_n/2,    0], 
                                              ])
        return verts
    
    def _generate_octagon_verts(self, centre: tuple[float, float], width: float, height: float) -> ndarray:
        # width and height sets the rectangle we inscribe the ocatgon in
        if width < height:
            print("Octagon generation impossible")
            return None
        verts =  np.array(centre) + np.array([
                                              [-self.w_n/2,         self.h_n/4], 
                                              [-self.w_n/4,         self.h_n/2], 
                                              [ self.w_n/4,         self.h_n/2], 
                                              [ self.w_n/2,         self.h_n/4], 
                                              [ self.w_n/2,        -self.h_n/4], 
                                              [ self.w_n/4,        -self.h_n/2], 
                                              [-self.w_n/4,        -self.h_n/2], 
                                              [-self.w_n/2,        -self.h_n/4], 
                                              ])
        return verts
    
    def _generate_parallelogram_verts(self, centre: tuple[float, float], width: float, height: float) -> ndarray:
        # width and height sets the rectangle we inscribe the parallelogram in
        if width < height:
            print("Octagon generation impossible")
            return None
        verts =  np.array(centre) + np.array([
                                              [-self.w_n/2 + self.h_n/np.tan(np.pi/2.4),       self.h_n/2], 
                                              [ self.w_n/2,                                  self.h_n/2], 
                                              [ self.w_n/2 - self.h_n/np.tan(np.pi/2.4),       -self.h_n/2], 
                                              [ -self.w_n/2,                                 -self.h_n/2], 
                                              ])
        return verts

    def update(self):
        self._draw_status(self.tree.root)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # self.fig.savefig("bt.png")


class GenericBehaviour(pt.behaviour.Behaviour):
    """
    Most of the behaviors are doing the same thing: they send a service requests, wait for answer and change the 
    blackboard accordingly the name of the behavior is also the name of the service 
    """
    def __init__(self, name, blackboard, logger, client=None):
        super().__init__(name)
        self.logger = logger
        self.logger.info("Init behaviour: " + self.name)
        self.blackboard = blackboard
        if client is None:
            self.ros_client = SendStrClient(name)
        else:
            self.ros_client = client

    def update(self):
        if self.blackboard.get(self.name) == "requested" or self.blackboard.get(self.name) == "running":    
            # self.logger.info("Update step start for behaviour: " + self.name + " current status: " + str(self.status))
            response = self.ros_client.send_request("")
            # self.logger.info(self.name + " response: " + response.ans)
            if response.ans == "success":
                new_status =  pt.common.Status.SUCCESS
            elif response.ans == "running":
                new_status =  pt.common.Status.RUNNING
                self.blackboard.set(self.name, "running")
            elif response.ans == "failure":
                new_status =  pt.common.Status.FAILURE
            else: 
                new_status = self.status
        elif self.blackboard.get(self.name) == "completed":
            new_status = pt.common.Status.SUCCESS
        elif self.blackboard.get(self.name) == "failed":
            new_status = pt.common.Status.FAILURE
        else:
            # self.logger.debug("Not supposed to run " + self.name)
            new_status = self.status
        # self.logger.info("Update step for behaviour: " + self.name + " end with status: " + str(new_status))
        return new_status

    def terminate(self, new_status):
        if new_status != pt.common.Status.INVALID:
            # if new_status != self.status:
            #     self.logger.info("Terminating: " + self.name + " with status: " + str(new_status) + "from status: " + str(self.status))
            if new_status == pt.common.Status.SUCCESS:
                self.status = new_status
                self.blackboard.set(self.name, "completed")
            elif new_status == pt.common.Status.FAILURE:
                self.status = new_status
                self.blackboard.set(self.name, "failed")

