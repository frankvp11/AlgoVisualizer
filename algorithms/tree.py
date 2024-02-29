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

    def traverse_recursive(self, x, y):
        
        if self.value:
            print("Value: ", self.value)
            # circle = Circle(x, y, 45, color='darkgray', transparency=0.9)
            # circle.give_outline(color='black', thickness=2)
            # text = Text(self.value, x-3, y, font_size=40)

        if self.left and self.left.value:
            
            # arrow = Arrow(x, y, x - 250, y + 250)
            self.left.traverse_recursive(x - 250, y + 250)
            

        if self.right and self.right.value:
            # arrow = Arrow(x, y, x + 250, y + 250)
            self.right.traverse_recursive(x + 250, y + 250)
