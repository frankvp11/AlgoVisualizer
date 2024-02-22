from nicegui import ui, app
import sys
import random
import time

sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2 import Rectangle, Text, ShapeCollection


class SVGContent():
    def __init__(self, polygons) -> None:
        self.content = polygons.to_svg()
        self.polygons = polygons
    
    def update_polygons(self, index):
        self.polygons.polygons[index].set_color("red")
        self.content = self.polygons.to_svg()




def make_row_of_numbers(numbers):
    rectangles = []
    textboxes = []

    for i in range(len(numbers)):
        rect = Rectangle.Rectangle(i*50, 0, 50, 50, "lightgray")
        rect.give_outline("black", 2)
        t = Text.Text(str(numbers[i]), i*50+15, 35  ,  "black", 20)
        rectangles.append(rect)
        textboxes.append(t)
    collection = ShapeCollection.ShapeCollection(rectangles + textboxes)
    collection.move_all(50, 0)
    return  collection



def start_timer(svgcontent, numbers, target=None):
    previous = [0]
    numbers2 = numbers
    target = numbers[2]
    def update_previous(svgcontent):
        nonlocal numbers2
        middle = len(numbers2) // 2

        if middle == target:
            timer.cancel()
            return
        if numbers2[middle] < target:
            numbers2 = numbers2[middle:]
            for i in range(middle):
                svgcontent.update_polygons(i)
        else:
            numbers2 = numbers2[:middle]
            for i in range(middle, 20):
                svgcontent.update_polygons(i)

        previous[0] += 1

    timer = ui.timer(2, lambda : update_previous(svgcontent))


def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
        ui.label("Binary Search Algorithm").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
    with ui.row():
        numbers= sorted([ random.randint(0, 100) for i in range(20)])

        image = ui.interactive_image(source="/static/binarytreesvg.svg").style("width: 2000px;")
        svgcontent=  SVGContent(make_row_of_numbers(numbers))

        image.bind_content_from(svgcontent, 'content')

    with ui.row():
        ui.button("Start animation", on_click=lambda e : start_timer(svgcontent, numbers))
