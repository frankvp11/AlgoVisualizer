from nicegui import ui, app
import networkx as nx
import sys
sys.path.append('/home/frank/Projects/Python/Algorithms')
from algorithms.node import Node
from algorithms.graph import Graph  
import random


async def add():

    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    graph = nx.MultiGraph()
    adjaceny_list = {
        1: [2, 3, 4, 5, 6],
        2: [1, 3, 9],
        3: [1, 2],
        4: [1, 5, 9],
        5: [1, 4, 6],
        6: [1, 5],
        7: [8, 9],
        8: [7, 9],
        9: [8, 7, 2, 4]
    }
    graph2 = Graph(adjacency_list=adjaceny_list)

    for key in adjaceny_list:
        graph.add_node(key)
        for val in adjaceny_list[key]:
            graph.add_edge(key, val)
    pos = nx.kamada_kawai_layout(graph)
    graph2.nodes = [graph2.add_node(key, 250+ 200*pos[key][0], 250+200*pos[key][1], id(key)) for key in adjaceny_list]
        

    @ui.refreshable
    async def stuff():
        for key in adjaceny_list:
            for val in adjaceny_list[key]:
                graph2.add_edge(key, val, id(key), id(val), weight=random.randint(1, 10))
        
        with ui.row().style("width: 100vw; "):
            with ui.column().style("width: 60vw; "):
                # graph2.dfs_animated(graph2.nodes[0])
                ui.button("Reset", on_click=lambda e : stuff.refresh())
                # ui.button("Dijkstras Algorithm", on_click=lambda e : )
                graph2.make_svg()
                image = ui.interactive_image("/static/dfssvg.svg").style("width: 100vw;")
                image.bind_content_from(graph2, 'content')
            with ui.column().style("width: 30vw; "):
                start_input = ui.input("Start", value=1)
                end_input = ui.input("End", value=7)
                async def call_dijkstras(e):
                    print("Running Dijkstras")
                    print("Start: ", start_input.value)
                    print("End: ", end_input.value)
                    await graph2.start_dijkstras2(int(start_input.value), int(end_input.value))
                ui.button("Run Dijkstras", on_click=call_dijkstras)


    await stuff()