

from nicegui import ui, app

from .heap_ds import Heap
import random

async def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    

    @ui.refreshable
    async def stuff():
        nums = [random.randint(1, 100) for i in range(10)]
        heap = Heap(nums)
        heap.build_heap()
        heap.content = heap.polygons.to_svg()
        root = heap.heap_to_binary_tree()
        root.traverse_recursive(750, 100)
        image = ui.interactive_image("/static/heaps.svg").style("width: 100vw;")
        print("length", len(root.shapes.polygons))
        print(root.shapes.polygons)
        image.content = root.shapes.to_svg()


    await stuff()