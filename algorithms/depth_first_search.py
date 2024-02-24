
import sys
import math
import random
import networkx as nx


sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.Arrow import Arrow
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Line import Line
from nicegui import ui, app



class Node:
    def __init__(self, data, **kwargs):
        self.data = data
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.polygon = ShapeCollection()
        self.id = kwargs.get('id', 0)

    def create_polygon(self):
        circle = Circle(self.x, self.y, 15, color='darkgray', transparency=0.9)
        circle.give_outline("black", thickness=1)
        text = Text(self.data, self.x, self.y, fontsize=10)
        self.polygon.add_polygon(circle)
        self.polygon.add_polygon(text)
        return self.polygon




class Graph:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.node_polygons = []
        self.edge_polygons = []
        self.polygons = []

    def add_node(self, data, x, y, id):
        node = Node(data, x=x, y=y, id=id)
        self.nodes.append(node)
        polygon = node.create_polygon()
        self.node_polygons.append(polygon.polygons[0])
        self.node_polygons.append(polygon.polygons[1])  # Assuming polygons[0] represents the circle
        return node

    def add_edge(self, start, end, start_id, end_id):
        self.edges.append((start, end))
        start_node = self.find_node(start, start_id)
        end_node = self.find_node(end, end_id)
        line = Line(start_node.x, start_node.y, end_node.x, end_node.y, color='black')
        self.edge_polygons.append(line)
    
    def find_node(self, data, id):
        for node in self.nodes:
            if node.data == data and node.id == id:
                return node
        return None

    def to_svg(self):
        collection = ShapeCollection()
        for polygon in self.edge_polygons:
            collection.add_polygon(polygon)
        for polygon in self.node_polygons:
            collection.add_polygon(polygon)
        return collection.to_svg()

def add():

    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")


            
    graph = nx.MultiGraph()
    graph2 = Graph()
    adjaceny_list = {
        1: [2, 3, 5],
        2: [3],
        3: [5],
        5: [6],
        6: [],
    }
    for key in adjaceny_list:
        graph.add_node(key)
        for val in adjaceny_list[key]:
            graph.add_edge(key, val)
    pos = nx.kamada_kawai_layout(graph)
    graph2.nodes = [graph2.add_node(key, 150+ 100*pos[key][0], 150+100*pos[key][1], id(key)) for key in adjaceny_list]
    
    for key in adjaceny_list:
        for val in adjaceny_list[key]:
            graph2.add_edge(key, val, id(key), id(val))

    for node in graph2.nodes:
        print(node.data, node.x, node.y, node.id)


    image = ui.interactive_image("/static/dfssvg.svg")
    image.content = graph2.to_svg()
