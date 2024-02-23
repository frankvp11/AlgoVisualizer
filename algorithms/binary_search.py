from nicegui import ui, app
import sys
import random
import time

sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2 import Rectangle, Text, ShapeCollection



timer = None


class SVGContent():
    def __init__(self, polygons) -> None:
        self.content = polygons.to_svg()
        self.polygons = polygons
    
    def update_polygons(self, index, color='red'):
        self.polygons.polygons[index].set_color(color)
        self.content = self.polygons.to_svg()




def make_row_of_numbers(numbers):
    rectangles = []
    textboxes = []

    for i in range(len(numbers)):
        rect = Rectangle.Rectangle(i*50, 0, 50, 50, color="lightgray")
        rect.give_outline("black", 2)
        t = Text.Text(str(numbers[i]), i*50+15, 35  ,  color="black", fontsize=20)
        rectangles.append(rect)
        textboxes.append(t)
    collection = ShapeCollection.ShapeCollection(rectangles + textboxes)
    collection.move_all(50, 0)
    return  collection



def start_timer(svgcontent, numbers, target=None):
    global timer
    previous = [0]
    numbers2 = numbers
    low  =0
    high = len(numbers2) - 1
    def update_previous(svgcontent):
        nonlocal numbers2, low, high
        middle = (low + high) // 2
        if numbers2[middle] == target:

            svgcontent.update_polygons(middle, color='green')
            timer.cancel()
            return
        elif numbers2[middle] < target:
            low = middle + 1
            for i in range(middle):
                svgcontent.update_polygons(i)
        else:
            for i in range(middle, len(numbers2)):
                svgcontent.update_polygons(i)
            high = middle - 1

        previous[0] += 1

    timer = ui.timer(2, lambda : update_previous(svgcontent))

def add():

    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")


    @ui.refreshable
    def stuff():
        global timer
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Binary Search Algorithm").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row():
            numbers= sorted(random.sample(range(1, 100), 20))
            target = numbers[2]
            def update_target(e):
                nonlocal target
                try:
                    target = int(e.value)
                except ValueError:
                    pass

            image = ui.interactive_image(source="/static/binsearchsvg.svg").style("width: 2000px;")
            svgcontent=  SVGContent(make_row_of_numbers(numbers))

            image.bind_content_from(svgcontent, 'content')
        with ui.row():
            ui.input("Target", value=target, on_change=lambda e: update_target(e))
        with ui.row():
            ui.button("Start animation", on_click=lambda e : start_timer(svgcontent, numbers, target))
        with ui.row():
            ui.button("Stop animation", on_click=lambda e : timer.deactivate())
        with ui.row():
            ui.button("Reset", on_click=lambda e : stuff.refresh())
    stuff()