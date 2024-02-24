
import sys
import math
from math import sqrt
import random
import networkx as nx


sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.Arrow import Arrow
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Line import Line
from ModelMaker.graphicsSVG2.Rectangle import Rectangle
from nicegui import ui, app



class Node:
    def __init__(self, data, **kwargs):
        self.data = data
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.id = kwargs.get('id', 0)
        self.neighbors = []
        self.circle = None
        self.text = None

    def create_polygon(self):
        self.circle = Circle(self.x, self.y, 15, color='darkgray', transparency=0.9)
        self.circle.give_outline("black", thickness=1)
        self.text = Text(self.data, self.x, self.y, fontsize=10)
        return ShapeCollection([self.circle, self.text])



class Graph:
    def __init__(self, **kwargs) -> None:
        self.nodes = []
        self.edges = []
        self.node_polygons = []
        self.edge_polygons = []
        self.queue_polygons = []
        self.polygons = ShapeCollection()
        self.adjacency_list = kwargs.get('adjacency_list', {})
        self.content = ""
        self.queue = []
        self.visited = [False] * 1000
        self.previous = None
        self.done = False
        self.timer = None
    

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

    def add_edge(self, start, end, start_id, end_id):
        self.edges.append((start, end))
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

    def start_timer(self):
        self.timer = ui.timer(3, self.bfs_animated)


def add():

    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")


            
    graph = nx.MultiGraph()
    graph2 = Graph()
    adjaceny_list = {
        1: [2, 3, 4, 5, 6],
        2: [3],
        3: [],
        4: [5],
        5: [6],
        6: [],
        7: [8],
        8: [9],
        9: [7, 2, 4]

        
    }
    for key in adjaceny_list:
        graph.add_node(key)
        for val in adjaceny_list[key]:
            graph.add_edge(key, val)
    pos = nx.kamada_kawai_layout(graph)
    graph2.nodes = [graph2.add_node(key, 250+ 200*pos[key][0], 250+200*pos[key][1], id(key)) for key in adjaceny_list]
    
    for key in adjaceny_list:
        for val in adjaceny_list[key]:
            graph2.add_edge(key, val, id(key), id(val))
            
    with ui.row():
        with ui.column().style("width: 100vw; "):
            graph2.bfs_animated(graph2.nodes[0])
            ui.button("Breadth First Search", on_click=lambda e : graph2.start_timer())
            graph2.make_svg()
            image = ui.interactive_image("/static/bfssvg.svg").style("width: 100vw;")
            image.bind_content_from(graph2, 'content')

