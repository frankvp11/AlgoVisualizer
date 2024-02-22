from nicegui import ui, app
import random
import sys
sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2 import Rectangle, Text, ShapeCollection, Circle, Polygon


timer = None

class BinaryTree:
    def __init__(self, value, x, y) -> None:
        self.value = value
        self.left = None
        self.right = None
        self.polygon = [Circle.Circle(x, y, 10), Text.Text(str(value), x, y-20, "black", 10)]

    def insert(self, value, x, y):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value, x-50, y+50)
            else:
                self.left.insert(value, x-50, y+50)
        else:
            if self.right is None:
                self.right = BinaryTree(value, x+50, y+50)
            else:
                self.right.insert(value, x+50, y+50)

def determine_binary_tree_depth_next_step_recursive(root, current_depth=1):
    if root is None:
        return
    else:
        yield root, current_depth
        if root.left:
            yield from determine_binary_tree_depth_next_step_recursive(root.left, current_depth + 1)
        if root.right:
            yield from determine_binary_tree_depth_next_step_recursive(root.right, current_depth + 1)


class SVGContent():
    def __init__(self, polygons) -> None:
        self.content = polygons.to_svg()
        self.polygons = polygons
    
    def update_polygons(self, polygon, color='red'):
        self.polygons.polygons[self.polygons.polygons.index(polygon)].set_color(color)
        # self.polygons.polygons[index].set_color(color)
        self.content = self.polygons.to_svg()



def make_tree_polygons(tree):
    if tree is None:
        return []
    else:
        return tree.polygon + make_tree_polygons(tree.left) + make_tree_polygons(tree.right)
        

def make_tree():
    num_nodes = random.randint(5, 10)
    # num_nodes = 5
    values = random.sample(range(1, 100), num_nodes)
    tree = BinaryTree(values[0], 100, 50)
    for value in values[1:]:
        tree.insert(value, 100, 50)

    polygons = make_tree_polygons(tree)    

    group = ShapeCollection.ShapeCollection(polygons)
    group.move_all(150, 0)
    return group, tree








    

def start_timer(tree, svgcontent):
    global timer
    index = [0]
    def timer_callback(steps, svgcontent):
        global timer
        index[0] += 1
        if index[0] >= len(steps):
            svgcontent.update_polygons(steps[index[0]-1][0].polygon[0], 'black')

            timer.cancel()
            return
        
        
        
        svgcontent.update_polygons(steps[index[0]][0].polygon[0], 'red')
        svgcontent.update_polygons(steps[index[0]-1][0].polygon[0], 'black')
        # step[0].polygon[0].set_color('red')
        print("Depth:", steps[index[0]][1])
        print("Tree:", steps[index[0]][0].value)

    steps = [s for s in determine_binary_tree_depth_next_step_recursive(tree)]
    timer = ui.timer(2, lambda : timer_callback(steps, svgcontent))


def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    
    
    @ui.refreshable
    def stuff():   

        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Depth of Binary Tree Algorithm").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            image = ui.interactive_image(source="/static/binarytreesvg.svg").style("width: 400px; height: 400px;")
            polygons, tree = make_tree()
            svgcontent = SVGContent(polygons)
            image.bind_content_from(svgcontent, 'content')
        with ui.row():
            ui.button("Refresh", on_click= lambda e : stuff.refresh())

        with ui.row():
            ui.button("Start timer", on_click= lambda e : start_timer(tree, svgcontent))
    stuff()