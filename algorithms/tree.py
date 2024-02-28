import sys

from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.Line import Line
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Arrow  import Arrow
class Tree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.shapes = ShapeCollection()

    def traverse_recursive(self, x, y):
        print(self.value)
        if self.value:
            circle = Circle(x, y, 15, color='darkgray', transparency=0.9)
            circle.give_outline(color='black', thickness=2)
            text = Text(self.value, x-3, y, fontsize=10)
            self.shapes.add_polygon(circle)
            self.shapes.add_polygon(text)
        if self.left:
            new_x = 0.9 * ( (x-100) - x) + x
            new_y = 0.9 * ( (y+100) - y) + y
            start_x  = 0.9 * (x - (x-100)) + (x-100)
            start_y  = 0.9 * (y - (y+100)) + (y+100)
            arrow = Arrow(start_x, start_y, new_x, new_y)
            self.shapes.polygons.extend(arrow.polygons)

            self.left.traverse_recursive(x-100, y+100)
        if self.right:
            new_x = 0.9 * ( (x+100) - x) + x
            new_y = 0.9 * ( (y+100) - y) + y
            start_x  = 0.9 * (x - (x+100)) + (x+100)
            start_y  = 0.9 * (y - (y+100)) + (y+100)
            arrow = Arrow(start_x, start_y, new_x, new_y)
            self.shapes.polygons.extend(arrow.polygons)
            self.right.traverse_recursive(x+100, y+100)

