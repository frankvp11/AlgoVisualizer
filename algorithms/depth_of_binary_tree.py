from nicegui import ui, app
import random
import sys
sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2 import Rectangle, Text, ShapeCollection, Circle, Polygon, Arrow


timer = None

class BinaryTree:
    def __init__(self, value, x, y) -> None:
        self.value = value
        self.left = None
        self.right = None
        self.polygon = [Circle.Circle(x, y, 10), Text.Text(str(value), x, y-20, color="black", fontsize=10), Text.Text("", x-20, y-20, color="red", fontsize=10)]
        self.x = x
        self.y = y

    def insert(self, value, x, y):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value, x-50, y+50)
                new_x = 0.9 * ( (x-50) - x) + x
                new_y = 0.9 * ( (y+50) - y) + y
                start_x  = 0.9 * (x - (x-50)) + (x-50)
                start_y  = 0.9 * (y - (y+50)) + (y+50)
                arrow = Arrow.Arrow(start_x, start_y, new_x, new_y)
                
                self.left.polygon.extend(arrow.polygons)
            else:
                self.left.insert(value, x-50, y+50)
        else:
            if self.right is None:
                self.right = BinaryTree(value, x+50, y+50)
                new_x = 0.9 * ( (x+50) - x) + x
                new_y = 0.9 * ( (y+50) - y) + y
                start_x  = 0.9 * (x - (x+50)) + (x+50)
                start_y  = 0.9 * (y - (y+50)) + (y+50)
                arrow = Arrow.Arrow(start_x, start_y, new_x, new_y)
                self.right.polygon.extend(arrow.polygons)
            else:
                self.right.insert(value, x+50, y+50)



class SVGContent():
    def __init__(self, polygons) -> None:
        self.polygons = polygons
        self.max_depth = 0
        self.current_depth = 0
        self.current_root = self.polygons.polygons[1].text
        self.polygons.polygons[0].set_color('red')
        self.content = polygons.to_svg()

    def update_polygons(self, polygon, color='red'):
        self.polygons.polygons[self.polygons.polygons.index(polygon)].set_color(color)
        self.content = self.polygons.to_svg()

    def start_timer(self, tree):
        global timer
        index = [0]
        def timer_callback(steps):
            global timer
            index[0] += 1
            if index[0] >= len(steps):
                self.update_polygons(steps[index[0]-1][0].polygon[0], 'black')

                timer.cancel()
                return
            
            
            self.update_polygons(steps[index[0]][0].polygon[0], 'red')
            self.update_polygons(steps[index[0]-1][0].polygon[0], 'black')
            
            self.current_depth = steps[index[0]][1]
            self.current_root = steps[index[0]][0].value
            self.max_depth = max(self.max_depth, steps[index[0]][1])
            steps[index[0]][0].polygon[2].text = self.current_depth


        steps = [s for s in self.determine_binary_tree_depth_next_step_recursive(tree)]
        timer = ui.timer(3, lambda : timer_callback(steps))

    def determine_binary_tree_depth_next_step_recursive(self, root, current_depth=1):
        if root is None:
            return
        else:
            yield root, current_depth
            if root.left:
                yield from self.determine_binary_tree_depth_next_step_recursive(root.left, current_depth + 1)
            if root.right:
                yield from self.determine_binary_tree_depth_next_step_recursive(root.right, current_depth + 1)



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
    group.move_all(120, 0)
    return group, tree



def implementation():
    markdown_text_2 = """
    
    class BinaryTree:
        def __init__(self, value) -> None:
            self.value = value
            self.left = None
            self.right = None
    
        def insert(self, value):
            if value < self.value:
                if self.left is None:
                    self.left = BinaryTree(value)
                else:
                    self.left.insert(value)
            else:
                if self.right is None:
                    self.right = BinaryTree(value)
                else:
                    self.right.insert(value)
    def determine_binary_tree_depth(root, current_depth=1):
        if root is None:
            return 0
        else:
            left = determine_binary_tree_depth(root.left, current_depth + 1)
            right = determine_binary_tree_depth(root.right, current_depth + 1)
            return max(left, right) + 1
    
        """

    markdown_text = """
    ```python
    class BinaryTree:
        def __init__(self, value) -> None:
            self.value = value
            self.left = None
            self.right = None
    
        def insert(self, value):
            if value < self.value:
                if self.left is None:
                    self.left = BinaryTree(value)
                else:
                    self.left.insert(value)
            else:
                if self.right is None:
                    self.right = BinaryTree(value)
                else:
                    self.right.insert(value)
    def determine_binary_tree_depth(root, current_depth=1):
        if root is None:
            return 0
        else:
            left = determine_binary_tree_depth(root.left, current_depth + 1)
            right = determine_binary_tree_depth(root.right, current_depth + 1)
            return max(left, right) + 1
    ```        
        """
    async def copy_code():
        ui.run_javascript(
            'navigator.clipboard.writeText(`' + markdown_text_2 + '`)')
        ui.notify('Copied to clipboard', type='positive', color='primary')
    ui.icon('content_copy', size='xs').on('click', copy_code, []).style(
        "position: relative; top: 5.5vw; left: 47.5vw;")

    ui.markdown(markdown_text).style(
        "width: 90%; height: fit-content; font-size: 8px; background-color: white; padding: 10px; border-radius: 10px; border: 1px solid black;")


def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    
    
    @ui.refreshable
    def stuff():   

        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Depth of Binary Tree Algorithm").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row().style("width: 100vw; height: 25vw;"):
            with ui.column().style("width: 35vw; height: 20vw; "):
                image = ui.interactive_image(source="/static/binarytreesvg.svg").style("width: 40vw; height: 20vw;")
                polygons, tree = make_tree()
                svgcontent = SVGContent(polygons)
                image.bind_content_from(svgcontent, 'content')
            with ui.column().style("width: 20vw;"):
                with ui.row():
                    with ui.column():
                        ui.label("Current Depth:") 
                    with ui.column():
                        text = ui.label("")
                        text.bind_text_from(svgcontent, 'current_depth')
                with ui.row():
                    with ui.column():
                        ui.label("Current Root:")
                    with ui.column():
                        text = ui.label("")
                        text.bind_text_from(svgcontent, 'current_root')
                with ui.row():
                    with ui.column():
                        ui.label("Max Depth = max (")

                    with ui.column():
                        text = ui.label("")
                        text.bind_text_from(svgcontent, 'current_depth')
                    with ui.column():
                        ui.label(",")
                    with ui.column():
                        text = ui.label("")
                        text.bind_text_from(svgcontent, 'max_depth')
                    with ui.column():
                        ui.label(")")
                    with ui.column():
                        ui.label("=")
                    with ui.column():
                        text = ui.label("")
                        text.bind_text_from(svgcontent, 'max_depth')
            with ui.column().style("width: 20vw; margin-left: 10vw;"):
                with ui.row():
                    ui.button("Refresh", on_click= lambda e : stuff.refresh())
                with ui.row():
                    ui.button("Start timer", on_click= lambda e : svgcontent.start_timer(tree))
    
    
        with ui.row().style("width: 100vw; align-items:center; justify-content:center;"):
            with ui.row().style("width: 60vw; background-color:lightgrey; border-radius: 10px; padding: 10px; margin-top: 10px; border: 1px solid black;"):
                with ui.column().style("width: 100vw; height: fit-content; text-align:center; align-items:center; justify-content:center;"):
                    ui.label("Depth of a Binary Tree").style("font-size: 1.5vw; font-weight: bold;")
                with ui.column():
                    ui.label("The depth of a binary tree is the number of edges from the root to the furthest leaf node.")
                ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                with ui.column():
                    ui.label("Consider a binary search tree (BST) as a hierarchical structure where each node has at most two children: a left child and a right child. The depth of a binary search tree refers to the length of the longest path from the root node to any leaf node.").style(
                        "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                with ui.column():
                    ui.label("To understand this concept further, let's break it down into steps:").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                with ui.column():
                    ui.label("1. Begin at the root node of the BST.").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                with ui.column():
                    ui.label("2. Traverse down the tree, moving to the left child or the right child depending on whether the value being searched for is smaller or larger than the current node's value.").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                with ui.column():
                    ui.label("3. Continue this process until reaching a leaf node, which has no children.").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                with ui.column():
                    ui.label("The depth of the tree is then determined by the length of this longest path.").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                with ui.column():
                    ui.label("Understanding the depth of a binary search tree is crucial for assessing its efficiency. A well-balanced tree, where the depth is minimized, allows for faster search operations. Conversely, an unbalanced tree with significant depth may result in slower search times.").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                with ui.column():
                    ui.label("Overall, the depth of a binary search tree plays a fundamental role in determining its performance characteristics, making it a key aspect to consider when analyzing or designing tree-based data structures.").style("font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                
                
                ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px;")
                with ui.column().style("text-align: left; width: 100vw; "):
                    ui.label("Implementation").style("font-size: 1.5vw; font-weight: bold;")
                
                with ui.column().style("text-align: left; width: 100vw; "):
                    implementation()
                    



    stuff()



