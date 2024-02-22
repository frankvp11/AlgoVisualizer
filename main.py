from nicegui import ui, app
import drawer


header = ui.header()

def toggle_drawer():
    left_drawer.toggle()

left_drawer = drawer.add()
with header:
        ui.button(icon='menu', on_click=toggle_drawer)

ui.run()