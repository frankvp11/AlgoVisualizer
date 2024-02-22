from nicegui import ui, app, APIRouter
import algorithms.depth_of_binary_tree


router = APIRouter(prefix='/algorithms')

app.include_router(router)







def show_page(event, page):

    print(page)
    
    # print(event)

def add():
    pages = ["Depth of Binary Tree", "Dijkstras", "Binary Search"]
    icons = ["/static/binarytree.png", "/static/dijkstras.png", "/static/binsearch.png"]
    targets = ["depth_of_binary_tree", "dijkstras", "binary_search"]
    for i in range(len(pages)):
        with ui.link(
                text="",
                target=f"/algorithms/{targets[i]}").style("text-decoration:none; color: black; display: flex; float: left;"):           
              with ui.card().on("click", lambda e, i=i: show_page(e, pages[i])):
                ui.label(pages[i])
                ui.image(icons[i]).style("width: 100px; height: 100px;")



@ui.page('/algorithms/depth_of_binary_tree')
def main():
    algorithms.depth_of_binary_tree.add()