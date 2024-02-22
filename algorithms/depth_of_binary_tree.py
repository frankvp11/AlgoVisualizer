from nicegui import ui, app
import random
import sys
sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2 import Rectangle, Text, ShapeCollection, Circle, Polygon



class BinaryTree:
    def __init__(self, value, x, y) -> None:
        self.value = value
        self.left = None
        self.right = None
        self.polygon = [Circle.Circle(x, y, 10), Text.Text(str(value), x, y-20, "black", 10)]

    def insert(self, value, x, y):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value, x-25, y+25)
            else:
                self.left.insert(value, x-25, y+25)
        else:
            if self.right is None:
                self.right = BinaryTree(value, x+25, y+25)
            else:
                self.right.insert(value, x+25, y+25)
    

class SVGContent():
    def __init__(self, polygons) -> None:
        self.content = polygons.to_svg()
        self.polygons = polygons
    
    def update_polygons(self, index, color='red'):
        self.polygons.polygons[index].set_color(color)
        self.content = self.polygons.to_svg()


def make_tree_polygons(tree):
    if tree is None:
        return []
    else:
        return tree.polygon + make_tree_polygons(tree.left) + make_tree_polygons(tree.right)
        

def make_tree():
    # num_nodes = random.randint(5, 10)
    num_nodes = 5
    values = random.sample(range(1, 100), num_nodes)
    tree = BinaryTree(values[0], 100, 50)
    for value in values[1:]:
        tree.insert(value, 100, 50)

    polygons = make_tree_polygons(tree)    

    group = ShapeCollection.ShapeCollection(polygons)
    group.move_all(0, 0)
    return group




def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    
    
    @ui.refreshable
    def stuff():   

        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Depth of Binary Tree Algorithm").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row():
            image = ui.interactive_image(source="/static/binarytreesvg.svg").style("width: 80vw; height: 80vh;")
            tree = make_tree()
            svgcontent = SVGContent(tree)
            image.bind_content_from(svgcontent, 'content')
        with ui.row():
            ui.button("Refresh", on_click= lambda e : stuff.refresh())

    stuff()