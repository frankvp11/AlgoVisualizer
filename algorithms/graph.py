
import asyncio
import heapq
import sys
import math
from math import sqrt
import random
import networkx as nx
sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.Arrow import Arrow
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Line import Line
from ModelMaker.graphicsSVG2.Rectangle import Rectangle
from algorithms.node import Node
from nicegui import ui, app

from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection

class Graph:
    def __init__(self, **kwargs) -> None:
        self.nodes = []
        self.edges = [[0 for _ in range(100)] for _ in range(100) ]
        self.node_polygons = []
        self.edge_polygons = []
        self.edge_polygons_weights = []
        self.queue_polygons = []
        self.stack_polygons = []
        self.polygons = None
        self.polygons = ShapeCollection()
        self.polygons.clear_all_polygons()
        self.adjacency_list = kwargs.get('adjacency_list', {})
        self.content = ""
        self.queue = []
        self.stack = []
        self.visited = [False] * 1000
        self.previous = None
        self.done = False
        self.timer = None

    def render_stack(self):
        for polygon in self.stack_polygons:
            self.polygons.remove_polygon(polygon)
        stack_text_ = Text("Stack", 0, 0, fontsize=40)
        stack_text_.scale(2)
        stack_text_.move(250, 20)
        self.stack_polygons = [stack_text_]
        
        for index, node in enumerate(self.stack):
            rect = Rectangle(500, 50+50*index, 50, 50, color='lightgray', transparency=1)
            rect.give_outline("black", thickness=1)
            text = Text(node.data, 510, 70+50*index, fontsize=20)
            self.stack_polygons.append(rect)
            self.stack_polygons.append(text)

    def render_queue(self):
        for polygon in self.queue_polygons:
            self.polygons.remove_polygon(polygon)
        queue_text_ = Text("Queue", 0, 0, fontsize=40)
        queue_text_.scale(2)
        queue_text_.move(250, 20)
        self.queue_polygons = [queue_text_]
        
        for index, node in enumerate(self.queue):
            rect = Rectangle(500, 50+50*index, 50, 50, color='lightgray', transparency=1)
            rect.give_outline("black", thickness=1)
            text = Text(node.data, 510, 70+50*index, fontsize=20)
            self.queue_polygons.append(rect)
            self.queue_polygons.append(text)

        

    def add_node(self, data, x, y, id):
        node = Node(data, x=x, y=y, id=id)
        self.nodes.append(node)
        polygon = node.create_polygon()
        self.node_polygons.append(polygon.polygons[0])
        self.node_polygons.append(polygon.polygons[1])  
        return node

    def add_edge(self, start, end, start_id, end_id, weight=0):
        if self.edges[end][start] == 0 and self.edges[start][end] == 0:
            self.edges[start][end] = ( weight)
        
        start_node = self.find_node(start, start_id)
        end_node = self.find_node(end, end_id)
        dx = end_node.x - start_node.x
        dy = end_node.y - start_node.y
        distance = sqrt(dx ** 2 + dy ** 2)        
        start_intersection_x = start_node.x + (dx / distance) * 15
        start_intersection_y = start_node.y + (dy / distance) * 15        
        end_intersection_x = end_node.x - (dx / distance) * 15
        end_intersection_y = end_node.y - (dy / distance) * 15        
        line = Line(start_intersection_x, start_intersection_y, end_intersection_x, end_intersection_y, color='black')
        if weight != 0 and self.edges[end][start] == 0:
            self.edges[end][start] = weight
            text = Text(str(weight), (start_intersection_x + end_intersection_x-5) / 2, (start_intersection_y + end_intersection_y-5) / 2, fontsize=20)
            self.edge_polygons_weights.append((start, end, text))
            self.edge_polygons.append(text)
        self.edge_polygons.append(line)
        start_node.neighbors.append(end_node)
        end_node.neighbors.append(start_node)
    
    def find_node(self, data, id):
        for node in self.nodes:
            if node.data == data and node.id == id:
                return node
        return None

    def make_svg(self):
        self.content = ""
        for polygon in self.stack_polygons:
            self.polygons.add_polygon(polygon)
        for polygon in self.queue_polygons:
            self.polygons.add_polygon(polygon)
        for polygon in self.edge_polygons:
            self.polygons.add_polygon(polygon)
        for polygon in self.node_polygons:
            self.polygons.add_polygon(polygon)
        
        self.content = self.polygons.to_svg()

    def bfs_one_call(self):
        if self.queue:
            current = self.queue.pop(0)
            self.polygons.polygons[self.polygons.polygons.index(current.circle)].color = "red"
            if self.previous:
                self.polygons.polygons[self.polygons.polygons.index(self.previous.circle)].color = "lightgray"

            self.previous  = current
            for neighbor in current.neighbors:
                if not self.visited[neighbor.data]:
                    self.visited[neighbor.data] = True
                    self.queue.append(neighbor)
            if self.queue == []:
                self.done = True

        else:
            self.previous = None
            self.queue = []
            self.visited = [False] * 1000
            return

        
    def bfs_animated(self, start=None):
        if self.done: 
            self.polygons.polygons[self.polygons.polygons.index(self.previous.circle)].color = "lightgray"
            self.make_svg()
            self.timer.cancel()
            return
        if self.queue == [] :
            self.queue = [start]
            self.visited[start.data] = True
            return
        
        self.render_queue()
        self.bfs_one_call()

        self.make_svg()


    def dfs_one_call(self):

        if self.stack:
            current = self.stack.pop()
            self.polygons.polygons[self.polygons.polygons.index(current.circle)].color = "red"
            if self.previous:
                self.polygons.polygons[self.polygons.polygons.index(self.previous.circle)].color = "lightgray"

            self.previous  = current
            for neighbor in current.neighbors:
                if not self.visited[neighbor.data]:
                    self.visited[neighbor.data] = True
                    self.stack.append(neighbor)
            if self.stack == []:
                self.done = True

        else:
            self.previous = None
            self.stack = []
            self.visited = [False] * 1000
            return
        
    def dfs_animated(self, start=None):
        if self.done: 
            self.polygons.polygons[self.polygons.polygons.index(self.previous.circle)].color = "lightgray"
            self.make_svg()
            self.timer.cancel()
            return
        if self.stack == [] :
            self.stack = [start]
            self.visited[start.data] = True
            return
        
        self.render_stack()
        self.dfs_one_call()

        self.make_svg()


    def update_text_on_edge(self, start, end, weight):
        for edge in self.edge_polygons_weights:
            if edge[0] == start and edge[1] == end:
                edge[2].text = weight
                edge[2].set_color("red")
                self.make_svg()

    def update_neighbor_node(self, neighbor):
        for node in self.nodes:
            if node.data == neighbor:
                node.circle.color = "red"
                # node.text.color = "red"
                self.make_svg()

    def update_previous_node(self, previous):
        for node in self.nodes:
            if node.data == previous or node in self.previous.neighbors:
                node.circle.color = "lightgray"
                # node.text.color = "lightgray"
                self.make_svg()
            
    def update_current_node(self, current):

        for node in self.nodes:
            
            if node.data == current.data:
                node.circle.color = "green"
                # node.text.color = "green"
                self.make_svg()
    def update_previous_neighbor(self, neighbor):
        for node in self.nodes:
            if node.data == neighbor:
                node.circle.color = "lightgray"
                # node.text.color = "lightgray"
                self.make_svg()

        

    async def start_dijkstras2(self, start, end):
        start_node = None
        distances = {}
        for node in self.nodes:
            if node.data == start:
                start_node = node
            distances[node.data] = math.inf
        distances[start] = 0
        predecessors = {}

        queue = [(0, start_node)]  # Priority queue ordered by distance

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_distance > distances[current_node.data]:
                continue  # Skip if already processed

            for neighbor in current_node.neighbors:
                distance = current_distance + self.edges[current_node.data][neighbor.data]
                if distance < distances[neighbor.data]:
                    distances[neighbor.data] = distance
                    predecessors[neighbor.data] = current_node.data
                    heapq.heappush(queue, (distance, neighbor))



        # print(distances)
        distances_2 = [ 0]  * (max(distances.keys())+1)
        for k in distances:
            distances_2[k] = distances[k]
        print(distances_2)
        print(predecessors)
        shortest_path = self.construct_shortest_path(start, end, predecessors)
        await self.animate_shortest_path(shortest_path)

    def construct_shortest_path(self, start, end, predecessors):
        shortest_path = []
        current_node = end
        while current_node != start:
            shortest_path.append(current_node)
            current_node = predecessors[current_node]
        shortest_path.append(start)
        shortest_path.reverse()
        return shortest_path
   
    async def animate_shortest_path(self, path):
        for index, node in enumerate(path):
            start_node = self.find_node(node, id(node))
            next_node = self.find_node(path[index+1], id(path[index+1])) if index < len(path) - 1 else None
            if next_node:
                start_node.circle.color = "green"
                next_node.circle.color = "green"
                self.make_svg()
                await asyncio.sleep(2)
            else:
                start_node.circle.color = "green"
                self.make_svg()
                await asyncio.sleep(2)
        await asyncio.sleep(3)
        for index, node in enumerate(path):
            start_node = self.find_node(node, id(node))
            next_node = self.find_node(path[index+1], id(path[index+1])) if index < len(path) - 1 else None
            if next_node:
                start_node.circle.color = "darkgray"
                next_node.circle.color = "darkgray"
            else:
                start_node.circle.color = "darkgray"
        self.make_svg()

    

    def start_timer_bfs(self):
        self.timer = ui.timer(3, self.bfs_animated)
    
    def start_timer_dfs(self):
        self.timer = ui.timer(3, self.dfs_animated)
