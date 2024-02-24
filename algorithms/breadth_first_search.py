
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
from algorithms.node import Node
from algorithms.graph import Graph
from nicegui import ui, app







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
        


    @ui.refreshable
    def stuff():
        for key in adjaceny_list:
            for val in adjaceny_list[key]:
                graph2.add_edge(key, val, id(key), id(val))
                
        with ui.row():
            with ui.column().style("width: 100vw; "):
                graph2.bfs_animated(graph2.nodes[0])
                ui.button("Breadth First Search", on_click=lambda e : graph2.start_timer_bfs())
                graph2.make_svg()
                image = ui.interactive_image("/static/bfssvg.svg").style("width: 100vw;")
                # print("Image content before binding in breadth_first_search.py")
                # print(image.content)
                image.bind_content_from(graph2, 'content')
                # print("Image content after binding in breadth_first_search.py")
                # print(image.content)
    stuff()
