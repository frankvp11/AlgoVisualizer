
from sklearn.datasets import make_blobs
from nicegui import ui
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio

from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from ModelMaker.graphicsSVG2.CustomPolygon import CustomPolygon
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Arrow import Arrow

from functools import partial

import matplotlib.pyplot as plt
rng = np.random.RandomState(1311)


class LatexElement:
    def __init__(self, equation):
        self.htlm_ele = ui.html(f'''
                <body> $${equation}$$ </body>
                ''')

    def update(self):
        # ui.update(self.htlm_ele)
        ui.run_javascript('MathJax.typeset()')


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ANN():
    def __init__(self, input_size=2, hidden_size1=4, hidden_size2=4, output_size=3):
        self.X, self.y = self.get_points()
        self.model = SimpleModel(
            input_size, hidden_size1, hidden_size2, output_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = 0
        self.graph = self.generate_graph()
        self.graph.move_all(100, 100)
        self.content = self.graph.to_svg()
        self.previous_contours = []
        self.current_epoch = 0
        self.timer_speed = 0.5
        self.timer = None
        self.model_graph = ShapeCollection()
        self.layer_1_shapes = []
        self.layer_2_shapes = []
        self.layer_3_shapes = []
        self.layer_4_shapes = []
        self.stage = "forward"
        self.create_model_graphic()
        self.parameters = 0

    def get_points(self):
        centers = [[-2.5, -2.5], [1.5, -1.5], [2.5, 2.5]]
        X, y = make_blobs(n_samples=2000, centers=centers, cluster_std=1.8)
        noise_strength = 0.5
        noise = np.random.normal(loc=0.0, scale=noise_strength, size=X.shape)
        noise_y = np.random.randint(0, 1, size=y.shape)
        X = X + noise
        y = y + noise_y
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y

    def start_timer(self):
        if self.timer_speed == None:
            self.timer_speed = 0.5
        self.timer = ui.timer(interval=self.timer_speed,
                              callback=self.run_epoch)

    def stop_timer(self):
        if self.timer == None:
            return
        if self.timer.active:

            self.timer.deactivate()
        else:
            self.resume_timer()

    def resume_timer(self):
        self.timer.activate()

    def end_timer(self):
        self.timer.cancel()

    def run_epoch(self):
        if self.timer_speed > 2:
            self.go_forward()

        self.current_epoch += 1
        self.optimizer.zero_grad()
        output = self.model(self.X)
        loss = self.loss_function(output, self.y)
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()
        self.update_boundaries()
        self.content = self.graph.to_svg()
        self.model_graph_image = self.model_graph.to_svg()
        if self.loss < 0.01:
            self.end_timer()
            print("Training complete")
        print("Running epoch", self.current_epoch)

    def go_forward(self):
        def layer1():
            for index in range(len(self.layer_1_shapes)):
                self.layer_1_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer2():
            for index in range(len(self.layer_1_shapes)):
                self.layer_1_shapes[index].set_color("black")
            for index in range(len(self.layer_2_shapes)):
                self.layer_2_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer3():
            for index in range(len(self.layer_2_shapes)):
                self.layer_2_shapes[index].set_color("black")
            for index in range(len(self.layer_3_shapes)):
                self.layer_3_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer4():
            for index in range(len(self.layer_3_shapes)):
                self.layer_3_shapes[index].set_color("black")
            for index in range(len(self.layer_4_shapes)):
                self.layer_4_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer5():
            for index in range(len(self.layer_4_shapes)):
                self.layer_4_shapes[index].set_color("black")
            self.model_graph_image = self.model_graph.to_svg()
            self.stage = "backward"

        def layer6():
            for index in range(len(self.layer_4_shapes)):
                self.layer_4_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer7():
            for index in range(len(self.layer_4_shapes)):
                self.layer_4_shapes[index].set_color("black")
            for index in range(len(self.layer_3_shapes)):
                self.layer_3_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer8():
            for index in range(len(self.layer_3_shapes)):
                self.layer_3_shapes[index].set_color("black")
            for index in range(len(self.layer_2_shapes)):
                self.layer_2_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer9():
            for index in range(len(self.layer_2_shapes)):
                self.layer_2_shapes[index].set_color("black")
            for index in range(len(self.layer_1_shapes)):
                self.layer_1_shapes[index].set_color("red")
            self.model_graph_image = self.model_graph.to_svg()

        def layer10():
            for index in range(len(self.layer_1_shapes)):
                self.layer_1_shapes[index].set_color("black")
            self.model_graph_image = self.model_graph.to_svg()

        self.stage = "forward"
        ui.timer(interval=0.05, callback=layer1, once=True)
        ui.timer(interval=2 * self.timer_speed /
                 10, callback=layer2, once=True)
        ui.timer(interval=3 * self.timer_speed /
                 10, callback=layer3, once=True)
        ui.timer(interval=4 * self.timer_speed /
                 10, callback=layer4, once=True)
        ui.timer(interval=5 * self.timer_speed /
                 10, callback=layer5, once=True)
        ui.timer(interval=6 * self.timer_speed /
                 10, callback=layer6, once=True)
        ui.timer(interval=7 * self.timer_speed /
                 10, callback=layer7, once=True)
        ui.timer(interval=8 * self.timer_speed /
                 10, callback=layer8, once=True)
        ui.timer(interval=9 * self.timer_speed /
                 10, callback=layer9, once=True)
        ui.timer(interval=10 * self.timer_speed /
                 10, callback=layer10, once=True)

    def generate_graph(self):
        points = []
        color_mapping = {0: "red", 1: "blue", 2: "green"}
        for (x, x2), color in zip(self.X, self.y):
            points.append(Circle(x.item()*10, x2.item()*10, 1,
                          color=color_mapping[color.item()]))
        return ShapeCollection(points)

    def update_boundaries(self):
        X = self.X.detach().numpy()
        y = self.y.detach().numpy()
        model = self.model
        color_mapping = {0: "red", 1: "blue", 2: "green"}

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        temp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        # print(temp)
        Z = model(temp)
        Z = Z.detach().numpy()
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        contours = plt.contourf(xx, yy, Z, alpha=0.5, levels=2)
        answers = []
        for i in range(len(contours.collections)):

            try:
                vertices = []
                for path in ((contours.collections[i].get_paths())[0].vertices):
                    vertices.append([path[0]*10, path[1]*10])

                polygon = CustomPolygon(
                    vertices=vertices, color=color_mapping[i], transparency=0.5)

                polygon.move(100, 100)
                answers.append(polygon)
            except:
                pass
        if self.previous_contours:
            for contour in self.previous_contours:
                self.graph.remove_polygon(contour)

        self.previous_contours = answers

        self.graph = ShapeCollection(answers + self.graph.polygons)
        self.content = self.graph.to_svg()

    def get_n_params(self):
        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        self.parameters = pp

    def create_model_graphic(self):
        self.model_graph = ShapeCollection(
            [Text("", 75, 50, color="black", size=10)])
        self.model_graph_image = self.model_graph.to_svg()

        total_height = 300-50
        space_each = total_height / (self.model.fc1.in_features+1)
        positions = [
            i*space_each for i in range(1, self.model.fc1.out_features+1)]

        for i in range(self.model.fc1.in_features):
            circ = Circle(25, positions[i], 10, color="black")
            self.layer_1_shapes.append(circ)
            self.model_graph.add_polygon(circ)
        # hidden layer 1
        total_height = 300-50
        space_each = total_height / (self.model.fc1.out_features+1)
        positions_2 = [
            i*space_each for i in range(1, self.model.fc1.out_features+1)]

        for i in range(self.model.fc1.out_features):
            circ = Circle(100, positions_2[i], 10, color="black")
            self.layer_2_shapes.append(circ)
            self.model_graph.add_polygon(circ)

        # hidden layer 2
        total_height = 300-50
        space_each = total_height / (self.model.fc2.out_features+1)
        positions_3 = [
            i*space_each for i in range(1, self.model.fc2.out_features+1)]
        for i in range(self.model.fc2.out_features):
            circl = Circle(175, positions_3[i], 10, color="black")
            self.layer_3_shapes.append(circl)
            self.model_graph.add_polygon(circl)

        # output layer
        total_height = 300-50
        space_each = total_height / (self.model.fc3.out_features+1)
        positions_4 = [
            i*space_each for i in range(1, self.model.fc3.out_features+1)]
        for i in range(self.model.fc3.out_features):
            circl = Circle(250, positions_4[i], 10, color="black")
            self.layer_4_shapes.append(circl)
            self.model_graph.add_polygon(circl)

        # connect the layers
        for i in range(self.model.fc1.in_features):
            for j in range(self.model.fc1.out_features):
                arr = Arrow(25, positions[i], 100,
                            positions_2[j],  color="black")
                # arr.move_all(0, 100)
                self.model_graph.add_polygon(arr)
        for i in range(self.model.fc1.out_features):
            for j in range(self.model.fc2.out_features):
                self.model_graph.add_polygon(
                    Arrow(100, positions_2[i], 175, positions_3[j]))
        for i in range(self.model.fc2.out_features):
            for j in range(self.model.fc3.out_features):
                self.model_graph.add_polygon(
                    Arrow(175, positions_3[i], 250, positions_4[j]))

        self.model_graph_image = self.model_graph.to_svg()


def create_model_svg():
    markdown_text = """```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleModel(nn.Module): 
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)      
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)      
        self.fc3 = nn.Linear(hidden_size2, output_size)   
                    
def forward(self, x):       
    x = F.relu(self.fc1(x))    
    x = F.relu(self.fc2(x))      
    x = self.fc3(x)      
    return x

model = SimpleModel(2, 4, 4, 3)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = 0
for epoch in range(1000):   
    optimizer.zero_grad()    
    output = model(X)   
    loss = loss_function(output, y)  
    loss.backward()    
    optimizer.step()   
    print(loss)   
```"""
    markdown_text_2 = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleModel(nn.Module): 
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)      
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)      
        self.fc3 = nn.Linear(hidden_size2, output_size)   
                    
def forward(self, x):       
    x = F.relu(self.fc1(x))    
    x = F.relu(self.fc2(x))      
    x = self.fc3(x)      
    return x

model = SimpleModel(2, 4, 4, 3)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = 0
for epoch in range(1000):   
    optimizer.zero_grad()    
    output = model(X)   
    loss = loss_function(output, y)  
    loss.backward()    
    optimizer.step()   
    print(loss)   

"""

    async def copy_code():
        ui.run_javascript(
            'navigator.clipboard.writeText(`' + markdown_text_2 + '`)')
        ui.notify('Copied to clipboard', type='positive', color='primary')
    ui.icon('content_copy', size='xs').on('click', copy_code, []).style(
        "position: relative; top: 6.5vw; left: 40.5vw;")

    ui.markdown(markdown_text).style(
        "width: 80%; height: fit-content; font-size: 8px; background-color: white; padding: 10px; border-radius: 10px; border: 1px solid black;")


def add():

    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")
    svgcontent = None
    

    
    def stuff():
        
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Artificial Neural Networks").style(
                "font-size: 3vw; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row():
            with ui.column().style("width: 200px; height: 200px;"):
                image = ui.interactive_image(source="/static/annsvg.svg")
                svgcontent = ANN()

                image.bind_content_from(svgcontent, 'content')
            with ui.column().style("width: 20vw; "):
                with ui.row():
                    with ui.column().style():
                        def generate_new_points():
                            svgcontent.__init__()

                        ui.button("Generate new graph", on_click=lambda: (generate_new_points())).style(
                            "font-size: 0.75vw; width: 8.5vw; height: 6.5vh; ")
                        ui.button("Start training", on_click=lambda: svgcontent.start_timer()).style(
                            "font-size: 0.75vw")
                        ui.button("Pause training", on_click=lambda: svgcontent.end_timer()).style(
                            "font-size: 0.75vw")
                        with ui.row():
                            ui.label("Timer speed:").style(
                                "font-size: 0.75vw;")
                            timer_speed_label = ui.label("1").style("font-size: 0.75vw;").bind_text_from(
                                svgcontent, 'timer_speed', lambda x: round((x if x != None else 0.5), 2))
                        timer_speed = ui.slider(min=0.1, max=10, step=0.1).style(
                            "font-size:10px; background-color:lightgrey;")
                        timer_speed.bind_value_to(svgcontent, 'timer_speed')

            def handle_layer_1_size_change(value, ann):
                print("Updating layer 1 size", value)
                try:
                    value = int(value)
                except ValueError:
                    value = 4
                if (value) == 0:
                    value = 1
                ann.model.fc1 = nn.Linear(2, value)
                ann.model.fc2 = nn.Linear(value, ann.model.fc2.out_features)
                ann.get_n_params()
                ann.create_model_graphic()

            def handle_layer_2_size_change(value, ann):
                print("Updating layer 2 size", value)
                try:
                    value = int(value)
                except ValueError:
                    value = 4
                if (value) == 0:
                    value = 1
                ann.model.fc2 = nn.Linear(ann.model.fc1.out_features, value)
                ann.model.fc3 = nn.Linear(value, ann.model.fc3.out_features)
                ann.get_n_params()
                ann.create_model_graphic()

            def handle_lr_change(value, ann):
                value = float(value)
                if (value) == 0:
                    value = 0.0001
                print("Updating learning rate", value)
                ann.optimizer = torch.optim.Adam(
                    ann.model.parameters(), lr=value)

            with ui.row():
                with ui.column().style("width: 10vw;"):

                    ui.input("Learning rate: ", value=0.001, on_change=lambda e: handle_lr_change(
                        e.value, svgcontent)).style("font-size: 0.75vw;")
                    ui.input("Hidden layer 1 size: ", value=4,  on_change=lambda e: handle_layer_1_size_change(
                        e.value, svgcontent)).style("font-size: 0.75vw;")
                    ui.input("Hidden layer 2 size: ", value=4,  on_change=lambda e: handle_layer_2_size_change(
                        e.value, svgcontent)).style("font-size: 0.75vw;")
                with ui.column():
                    # show the modele here
                    with ui.column():
                        ui.label().bind_text_from(svgcontent, 'stage')
                    with ui.column():
                        model_image = ui.interactive_image(
                            "/static/annsvg2.svg").style("width: 20vw; height: 20vw;")
                        model_image.bind_content_from(
                            svgcontent, 'model_graph_image')
                with ui.column():
                    with ui.row():
                        with ui.column():
                            ui.label("Loss:").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                        with ui.column():
                            loss = ui.label("").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                            loss.bind_text_from(svgcontent, 'loss')
                    with ui.row():
                        with ui.column():
                            ui.label("Epoch:").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                        with ui.column():
                            epoch = ui.label("").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                            epoch.bind_text_from(svgcontent, 'current_epoch')
                    with ui.row():
                        with ui.column():
                            ui.label("Parameters:").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                        with ui.column():
                            svgcontent.get_n_params()
                            parameters = ui.label("").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                            parameters.bind_text_from(svgcontent, 'parameters')
                    with ui.row():
                        with ui.column():
                            ui.label("Optimizer:").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                        with ui.column():
                            optimizer = ui.label("Adam").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                    with ui.row():
                        with ui.column():
                            ui.label("Loss function:").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")
                        with ui.column():
                            loss_function = ui.label("Cross Entropy").style(
                                "font-size: 0.75vw;  margin-bottom: 20px; justify-content:center;")

            with ui.row().style("width: 100vw; justify-content:center; align-items:center; height: fit-content;"):
                with ui.row().style("width: 60vw; justify-content:center; align-items:center; border: 1px solid black; background-color: lightgrey; border-radius: 10px; padding: 10px; overflow-y: visible; min-height: fit-content;"):

                    with ui.column().style("display: flex; justify-content: center; align-items: center; width: 100vw; "):
                        ui.label("How ANNs Work:").style(
                            "font-size: 2.5vw;  text-align:center; ")

                    with ui.column().style("width: 100vw; "):
                        ui.label("Artificial neural networks (ANNs) emulate the functionality of the human brain in computational models. They consist of interconnected nodes, each executing basic mathematical operations. The output of each node depends on its operation and a unique set of parameters. Through the interconnection of these nodes and precise parameter tuning, ANNs can effectively learn and compute highly intricate functions. ").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                    with ui.column().style("width: 100vw; "):
                        ui.label("The human brain predominantly consists of neurons, minute cells that adapt to emit electrical and chemical signals based on specific functions. There are approximately 100 billion neurons in the human brain, roughly 15 times the global population. On average, each neuron is linked to around 10,000 other neurons, resulting in a staggering 1 quadrillion connections between neurons.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                    with ui.row().style("width: 60vw; "):
                        with ui.column():
                            ui.image("/static/neuron.png").style(
                                "width: 20vw; height: 20vw; margin-top: 10px;")
                        with ui.column():
                            ui.label("Given that individual neurons aren't inherently capable of intricate computations, the immense number of neurons and connections is believed to underlie the brain's computational prowess. Despite the brain containing thousands of neuron types, artificial neural networks (ANNs) typically simplify their models by emulating only one type.").style(
                                "font-size: 1vw; justify-content:left; margin-top:3vh; font-family: 'Lucida Console', 'Courier New', monospace; width: 20vw;")

                    # ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                    with ui.column().style("width: 100vw; "):
                        ui.html("Neurons operate by firing when they receive sufficient input from connected neurons. This firing behavior is often represented by an <strong>activation</strong> function, where inputs below a certain threshold don't trigger firing, while those surpassing it do. Hence, neurons exhibit an all-or-nothing firing pattern, where they either fire or remain inactive.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    # ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                    with ui.column().style("width: 100vw; "):
                        ui.label("From a neuron's perspective, its connections can generally be categorized into incoming and outgoing connections. Incoming connections serve as the neuron's input, while outgoing connections transmit its output. Consequently, neurons receiving incoming connections from other neurons treat their outputs as inputs. This iterative process of transforming outputs into inputs contributes to the brain's computational power by creating highly complex functions through the composition of activation functions.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    with ui.column().style("width: 100vw; "):
                        ui.label("Interestingly, not all incoming connections are equal; some are stronger than others, providing more significant input to a neuron. As neurons fire when they receive input surpassing a certain threshold, these strong connections play a more influential role.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")

                    ui.column().style("width: 60vw; background-color: black; height: 1px; margin-top: 10px; margin-bottom: 10px;")
                    with ui.column().style("width: 100vw; "):
                        ui.label("Selecting an appropriate error function is pivotal in the training of artificial neural networks (ANNs). This involves defining an error function E over a set of input-output pairs  ").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        LatexElement(
                            "X = \{(x_1, y_1), .... , (x_N, y_N)\}").update()
                        ui.label("such that E(x,y) is minized when the output y approximates the target output y for all inputs x. Commonly used error functions include the mean squared error (MSE) for regression problems and cross-entropy for classification problems.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("To minimize the error function E(X,θ) with respect to the parameters θ, techniques like gradient descent are employed. Gradient descent iteratively updates the parameters by moving in the direction of the negative gradient of the error function. This process continues until a local minimum is reached or the gradient converges sufficiently.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("Gradient descent offers the advantage of being applicable to online learning, where parameters are updated incrementally as new input-output pairs arrive. It can also mimic batch learning if the step size η is appropriately chosen. However, it should be noted that gradient descent may converge to local minima due to its local nature. Despite this, recent research suggests that this is not a significant issue for ANNs, as most local minima are evenly distributed and similar in magnitude for large networks.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        ui.label("A critical breakthrough in ANN training came with the development of backpropagation in the mid-1980s. Backpropagation enables the calculation of gradients with respect to an ANN's parameters, making training feasible even for complex networks with numerous nodes and layers. This method revolutionized ANN research, facilitating the training of sophisticated models previously deemed mathematically intractable.").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                        
                    with ui.column().style("width: 100vw; "):

                        ui.label("Below you can see the code that I used to make and train this model:").style(
                            "font-size: 1vw; justify-content:center; margin-top:-10px; font-family: 'Lucida Console', 'Courier New', monospace;")
                    with ui.column().style("width: 100vw; "):
                        create_model_svg()
    stuff()
