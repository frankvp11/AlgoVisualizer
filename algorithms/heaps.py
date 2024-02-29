

from nicegui import ui, app

from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection

from .heap_ds import Heap
import random

async def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    

    @ui.refreshable
    async def stuff():
        nums = random.sample(range(1, 100), 5)
        print(nums)
        heap = Heap(nums)
        heap.build_heap()
        heap.content = heap.polygons.to_svg()
        root = heap.heap_to_binary_tree()
        # root.traverse_recursive(750, 100)
        
        with ui.row().style("width: 100vw; "):
            with ui.column().style("width: 30vw; "):
                image = ui.interactive_image("/static/heaps.svg")#.style("width: 30vw;")
                image.bind_content_from(heap, 'content')
            with ui.column().style("width: 30vw; "):



                def thing():
                    heap.extract_max()
                    image.content = ""
                    # heap.polygons = []
                    new_root = heap.heap_to_binary_tree()
                    
                    
                            
                    image.content = heap.polygons.to_svg()

                def add_value_to_heap():
                    heap.insert(int(add_value.value))
                    image.content = ""
                    new_root = heap.heap_to_binary_tree()
                    image.content = heap.polygons.to_svg()

                ui.button("Extract max", on_click=lambda e : thing())
                add_value = ui.input("Enter a number")
                add_value.on("keydown.enter", lambda e : add_value_to_heap())
    await stuff()