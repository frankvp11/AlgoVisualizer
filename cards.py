from nicegui import ui, app, APIRouter
import algorithms.depth_of_binary_tree
import algorithms.dijkstras
import algorithms.binary_search
import algorithms.artificialneuralnetwork
import algorithms.depth_first_search
import algorithms.breadth_first_search
router = APIRouter(prefix='/algorithms')

app.include_router(router)








def add():
    pages = ["Depth of Binary Tree", "Dijkstras", "Binary Search", 'Artificial Neural Network', "Depth First Search", "Breadth First Search"]
    icons = ["/static/binarytree.png", "/static/dijkstras.png", "/static/binsearch.png", '/static/ann.png', "/static/dfs.png", "/static/bfs.png"]
    targets = [depth_of_binary_tree, dijkstras, binary_search, artificialneuralnetwork, depth_first_search, breadth_first_search]
    with ui.row():
        for i in range(len(pages)):
            with ui.link(
                text="",
                target=targets[i]).style("text-decoration:none; color: black; display: flex; float: left;"):           
              with ui.card():
                ui.label(pages[i])
                ui.image(icons[i]).style("width: 100px; height: 100px;")



@ui.page('/algorithms/depth_of_binary_tree')
def depth_of_binary_tree():
    algorithms.depth_of_binary_tree.add()

@ui.page('/algorithms/dijkstras')
def dijkstras():
    algorithms.dijkstras.add()

@ui.page('/algorithms/binary_search')
def binary_search():
    algorithms.binary_search.add()

@ui.page('/algorithms/artificialneuralnetwork')
def artificialneuralnetwork():
    algorithms.artificialneuralnetwork.add()


@ui.page("/algorithms/depth_first_search")
def depth_first_search():
    app.storage.clear()
    algorithms.depth_first_search.add()


@ui.page("/algorithms/breadth_first_search")
def breadth_first_search():
    app.storage.clear()
    
    algorithms.breadth_first_search.add()
