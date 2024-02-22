from nicegui import ui, app
import cards


app.add_static_files("/static", 'static')

header = ui.header()
cards.add()



ui.run()