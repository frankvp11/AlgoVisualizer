from nicegui import ui, app, APIRouter
import algorithms.depth_of_binary_tree
import algorithms.dijkstras
import algorithms.binary_search
import algorithms.artificialneuralnetwork
router = APIRouter(prefix='/algorithms')

app.include_router(router)









def add():
    pages = ["Depth of Binary Tree", "Dijkstras", "Binary Search", 'Artificial Neural Network']
    icons = ["/static/binarytree.png", "/static/dijkstras.png", "/static/binsearch.png", '/static/ann.png']
    targets = ["depth_of_binary_tree", "dijkstras", "binary_search", 'artificialneuralnetwork']
    with ui.row():
        for i in range(len(pages)):
            with ui.link(
                text="",
                target=f"/algorithms/{targets[i]}").style("text-decoration:none; color: black; display: flex; float: left;"):           
              with ui.card():
                ui.label(pages[i])
                ui.image(icons[i]).style("width: 100px; height: 100px;")



@ui.page('/algorithms/depth_of_binary_tree')
def main():
    algorithms.depth_of_binary_tree.add()

@ui.page('/algorithms/dijkstras')
def main():
    algorithms.dijkstras.add()

@ui.page('/algorithms/binary_search')
def main():
    algorithms.binary_search.add()

@ui.page('/algorithms/artificialneuralnetwork')
def main():
    algorithms.artificialneuralnetwork.add()

