from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Line import Line
from ModelMaker.graphicsSVG2.Rectangle import Rectangle
from nicegui import ui, app
from ModelMaker.graphicsSVG2.Circle import Circle

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
