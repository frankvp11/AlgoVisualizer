from nicegui import ui, app


def show_page(event, page):

    print(page)
    print(event)

def add():
    pages = ["Depth of Binary Tree", "Dijkstras", "Binary Search"]
    icons = ["/static/binarytree.png", "/static/dijkstras.png", "/static/binsearch.png"]
    for i in range(len(pages)):
        with ui.card().on("click", lambda e, i=i: show_page(e, pages[i])):
            ui.label(pages[i])
            ui.image(icons[i]).style("width: 100px; height: 100px;")