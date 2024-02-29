from nicegui import ui, app
import sys
import random
import time

sys.path.append('/home/frank/Projects/Python/Algorithms')
from ModelMaker.graphicsSVG2.Rectangle import Rectangle
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection



timer = None


class SVGContent():
    def __init__(self, numbers) -> None:
        self.content = ""
        self.polygons = []
        self.numbers = numbers
        self.target = None
        self.low = 0
        self.high = len(self.numbers) - 1
        self.middle = None
        self.counter =0 
        self.timer = None
        self.make_row_of_numbers()
    
    def update_polygons(self, index, color='red'):
        self.polygons.polygons[index].set_color(color)
        self.content = self.polygons.to_svg()


    def make_row_of_numbers(self):
        rectangles = []
        textboxes = []

        for i in range(len(self.numbers)):
            rect = Rectangle(i*75, 0, 75, 75, color="lightgray")
            rect.give_outline("black", 2)
            t = Text(str(self.numbers[i]), i*75+15, 45  ,  color="black", fontsize=20)
            rectangles.append(rect)
            textboxes.append(t)
        collection = ShapeCollection(rectangles + textboxes)
        collection.move_all(100, 0)
        self.polygons = collection
        self.content = collection.to_svg()


    def update_previous(self):
            self.middle = (self.low + self.high) // 2
            if self.numbers[self.middle] == self.target:

                self.update_polygons(self.middle, color='green')
                if self.timer:
                    self.timer.cancel()
                return
            elif self.numbers[self.middle] < self.target:
                self.low = self.middle + 1
                for i in range(self.middle):
                    self.update_polygons(i)
            else:
                for i in range(self.middle, len(self.numbers)):
                    self.update_polygons(i)
                self.high = self.middle - 1

            self.counter += 1


    def start_timer(self):
        

        self.timer = ui.timer(2, lambda : self.update_previous())


def do_comparison(x, nums, middle):
    if not middle:
        return ""
    if x < nums[middle]:
        return f"Since {x} < {nums[middle]}, we take the lower half of the list"
    elif x > nums[middle]:
        return f"Since {x} > {nums[middle]} we take the upper half of the list"
    else:
        return f"Since {x} == {nums[middle]} we have found the target!"

def implementation():
    markdown_text_2 = """
    def binary_search(arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0

        while low <= high:
            mid = (high + low) // 2

            if arr[mid] < x:
                low = mid + 1

            elif arr[mid] > x:
                high = mid - 1

            else:
                return mid

        return -1
    
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = 5
    binary_search(arr, x)


    """
    markdown_text = """
    ```python
    def binary_search(arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0

        while low <= high:
            mid = (high + low) // 2

            if arr[mid] < x:
                low = mid + 1

            elif arr[mid] > x:
                high = mid - 1

            else:
                return mid

        return -1

    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = 5
    binary_search(arr, x)
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
        global timer
        with ui.row().style("width: 95vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Binary Search Algorithm").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row():
            numbers= sorted(random.sample(range(1, 100), 20))
            target = numbers[2]
            svgcontent=  SVGContent(numbers)
            with ui.column().style("width: 80vw; height: 30vw; justify-content:center; text-align:center; align-items:center;"):
                target_label = ui.label(f"Searching for: {target}").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
                target_label.bind_text_from(svgcontent, "target", backward=lambda x: "Searching for: " + str(x)).style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:right;")
                image = ui.interactive_image(source="/static/binsearchsvg.svg").style("width: 85vw;")
                image.bind_content_from(svgcontent, 'content')
                current_comparison_text = ui.label().style("font-size: 20px; font-weight: bold; margin-bottom: 20px; ")
                current_comparison_text.bind_text_from(svgcontent, "target", backward=lambda x: do_comparison(x, svgcontent.numbers, svgcontent.middle))

        
            # with ui.row().style("width: 35vw;"):
            with ui.column().style("width: 10vw; "):
                with ui.row():
                    ui.input("Target", value=target).bind_value_to(svgcontent, "target", forward= lambda x : int(x)).style("font-size: 0.75vw; ")
                with ui.row():
                    ui.button("Start", on_click=lambda e : svgcontent.start_timer()).style("font-size: 0.75vw; ")
                with ui.row():
                    ui.button("Stop animation", on_click=lambda e : svgcontent.timer.cancel()).style("font-size: 0.75vw; ")
                with ui.row():
                    ui.button("Step through", on_click=lambda e : svgcontent.update_previous()).style("font-size: 0.75vw; ")
                with ui.row():
                    ui.button("Reset", on_click=lambda e : stuff.refresh()).style("font-size: 0.75vw; ")
    
            # with ui.column().style("width: 50vw;"):
            #     with ui.row():
            
            
            with ui.row().style("width: 100vw; justify-content:center; align-items:center; height: fit-content;"):
                with ui.row().style("width: 60vw; justify-content:center; align-items:center; border: 1px solid black; background-color: lightgrey; border-radius: 10px; padding: 10px; overflow-y: visible; min-height: fit-content;"):

                    with ui.column().style("display: flex; justify-content: center; align-items: center; width: 100vw; "):
                        ui.label("How Binary Search Works:").style(
                            "font-size: 2.5vw;  text-align:center; ")

                    with ui.column().style("width: 100vw; "):
                        ui.label("The binary search algorithm efficiently seeks a specific element within a sorted list. For instance, in a scenario where a teacher needs to determine if any student scored 80 in a class with sorted test scores, binary search swiftly provides an answer. It operates by continually dividing the search space in half, narrowing down the potential location of the desired value. Binary search not only identifies the presence and position of an element within a list but also determines if it's absent altogether.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    with ui.column().style("width: 100vw; "):
                        ui.label("To execute binary search, the algorithm first compares the target value with the middle element of the array. If the target is greater, it disregards the left half of the list; if lesser, it discards the right half. This process iterates until the middle element matches the target or indicates that the element isn't present in the list.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                    ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                    with ui.column().style("width: 100vw; "):
                        ui.label("In essence, the algorithm operates as follows:").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("Begin by identifying the middle element of a sorted list. This is achieved by calculating the floor value of (low + high) / 2, where low represents the lowest index and high represents the highest index in the list. For instance, in the list [1, 2, 3, 4], the middle is 2 (index 1), and in [1, 2, 3, 4, 5], it is 3 (index 2).").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("Next, compare the middle element with the target value.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("If the target matches the middle element, indicate that the element exists in the list (optionally, return its index).").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("If the target is less than the middle element, discard all elements to the right (including the middle) and repeat the process with the narrowed search space.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("If the target is greater than the middle element, discard all elements to the left (including the middle) and repeat the process with the narrowed search space.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("It's important to note that binary search requires a pre-sorted list; attempting it on an unsorted list will result in failure. In such cases, a linear search may be more suitable.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                    with ui.column().style("width: 100vw; "):

                        ui.label("Below you can see the code that I used to run this algorithm:").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    with ui.column().style("width: 100vw; "):
                        implementation()

    stuff()







