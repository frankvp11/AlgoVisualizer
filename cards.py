from nicegui import ui, app, APIRouter
import algorithms.depth_of_binary_tree
import algorithms.dijkstras
import algorithms.binary_search
import algorithms.artificialneuralnetwork
import algorithms.depth_first_search
import algorithms.breadth_first_search
import algorithms.heaps

router = APIRouter(prefix='/algorithms')

app.include_router(router)


def add():
    pages = ["Depth of Binary Tree", "Dijkstras", "Binary Search",
             'Artificial Neural Network', "Depth First Search", "Breadth First Search", 'heaps']
    icons = ["/static/binarytree.png", "/static/dijkstras.png", "/static/binsearch.png",
             '/static/ann.png', "/static/dfs.png", "/static/bfs.png", "/static/heaps.png"]
    targets = [depth_of_binary_tree, dijkstras, binary_search,
               artificialneuralnetwork, depth_first_search, breadth_first_search, heaps]
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
async def dijkstras():
    await algorithms.dijkstras.add()


@ui.page('/algorithms/binary_search')
def binary_search():
    algorithms.binary_search.add()


@ui.page('/algorithms/artificialneuralnetwork')
def artificialneuralnetwork():
    head_html = '''
                <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                <script type="text/javascript" id="MathJax-script" async
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
                </script>
                '''
    ui.add_head_html(head_html)
    algorithms.artificialneuralnetwork.add()


@ui.page("/algorithms/depth_first_search")
def depth_first_search():
    app.storage.clear()
    algorithms.depth_first_search.add()


@ui.page("/algorithms/breadth_first_search")
def breadth_first_search():
    app.storage.clear()

    algorithms.breadth_first_search.add()


@ui.page('/algorithms/heaps')
async def heaps():
    await algorithms.heaps.add()
