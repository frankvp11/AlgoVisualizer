
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


import matplotlib.pyplot as plt
rng = np.random.RandomState(1311)

from sklearn.datasets import make_blobs
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

class ANN():
    def __init__(self, input_size=2, hidden_size1=4, hidden_size2=4, output_size=3):
        self.X, self.y = self.get_points()
        self.model = SimpleModel(input_size, hidden_size1, hidden_size2, output_size)
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
        self.timer = ui.timer(interval=self.timer_speed, callback=self.run_epoch)

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
                # self.model_graph.run_method(self.layer_1_shapes[index], "set_color", "red")
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
        ui.timer(interval=2 * self.timer_speed / 10, callback=layer2, once=True)
        ui.timer(interval=3 * self.timer_speed / 10 ,callback=layer3, once=True)
        ui.timer(interval=4 * self.timer_speed / 10, callback=layer4, once=True)
        ui.timer(interval=5  *self.timer_speed / 10, callback=layer5, once=True)
        ui.timer(interval=6 * self.timer_speed / 10, callback=layer6, once=True)
        ui.timer(interval=7 * self.timer_speed / 10, callback=layer7, once=True)
        ui.timer(interval=8 * self.timer_speed / 10, callback=layer8, once=True)
        ui.timer(interval=9 * self.timer_speed / 10, callback=layer9, once=True)
        ui.timer(interval=10 * self.timer_speed / 10, callback=layer10, once=True)

        

    def generate_graph(self):
        points = []
        color_mapping = {0: "red", 1: "blue", 2: "green"}
        for (x, x2), color in zip(self.X, self.y):
            points.append(Circle(x.item()*10, x2.item()*10, 1, color=color_mapping[color.item()])) 
        return ShapeCollection(points)
    
    def update_boundaries(self):
        X = self.X.detach().numpy()
        y = self.y.detach().numpy()
        model = self.model
        color_mapping = {0: "red", 1: "blue", 2: "green"}

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        temp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        # print(temp)
        Z = model(temp)
        Z = Z.detach().numpy()
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)        
        
        contours = plt.contourf(xx, yy, Z, alpha=0.5, levels=2)
        answers = []
        for i in range(len(contours.collections)): # 
            
            try:
                vertices = []
                for path in ((contours.collections[i].get_paths())[0].vertices):
                    vertices.append([path[0]*10, path[1]*10])

                polygon = CustomPolygon(vertices=vertices, color=color_mapping[i], transparency=0.5)
                
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





    def create_model_graphic(self):
        self.model_graph = ShapeCollection([Text("", 75, 50, color="black", size=10)])
        self.model_graph_image = self.model_graph.to_svg()

        total_height = 300-50
        space_each = total_height / (self.model.fc1.in_features+1)
        positions = [i*space_each for i in range(1, self.model.fc1.out_features+1)] 

        for i in range(self.model.fc1.in_features):
            circ = Circle(25, positions[i], 10, color="black")
            self.layer_1_shapes.append(circ)
            self.model_graph.add_polygon(circ)
        ## hidden layer 1
        total_height = 300-50
        space_each = total_height / (self.model.fc1.out_features+1)
        positions_2 = [i*space_each for i in range(1, self.model.fc1.out_features+1)] 

        for i in range(self.model.fc1.out_features):
            circ = Circle(100, positions_2[i], 10, color="black")
            self.layer_2_shapes.append(circ)
            self.model_graph.add_polygon(circ)

        ## hidden layer 2
        total_height = 300-50
        space_each = total_height / (self.model.fc2.out_features+1)
        positions_3 = [i*space_each for i in range(1, self.model.fc2.out_features+1)]
        for i in range(self.model.fc2.out_features):
            circl = Circle(175, positions_3[i], 10, color="black")
            self.layer_3_shapes.append(circl)
            self.model_graph.add_polygon(circl)

        
        ## output layer
        total_height = 300-50
        space_each = total_height / (self.model.fc3.out_features+1)
        positions_4 = [i*space_each for i in range(1, self.model.fc3.out_features+1)]
        for i in range(self.model.fc3.out_features):
            circl = Circle(250, positions_4[i], 10, color="black")
            self.layer_4_shapes.append(circl)
            self.model_graph.add_polygon(circl)
        
        ## connect the layers
        for i in range(self.model.fc1.in_features):
            for j in range(self.model.fc1.out_features):
                arr = Arrow(25, positions[i], 100, positions_2[j],  color="black")
                # arr.move_all(0, 100)
                self.model_graph.add_polygon(arr)
        for i in range(self.model.fc1.out_features):
            for j in range(self.model.fc2.out_features):
                self.model_graph.add_polygon(Arrow(100, positions_2[i], 175, positions_3[j]))
        for i in range(self.model.fc2.out_features):
            for j in range(self.model.fc3.out_features):
                self.model_graph.add_polygon(Arrow(175, positions_3[i], 250, positions_4[j]))

        self.model_graph_image = self.model_graph.to_svg()



def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")


    @ui.refreshable
    def stuff():
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Artificial Neural Networks").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row():
            with ui.column().style("width: 300px; height: 300px;"):
                image = ui.interactive_image(source="/static/annsvg.svg")
                svgcontent= ANN()
                
                
                image.bind_content_from(svgcontent, 'content')
            with ui.column().style("width: 20vw; "):
                with ui.row():
                    with ui.column():
                        ui.button("Generate new graph", on_click=lambda: (stuff.refresh()).style("font-size: 10px"))
                    with ui.column():
                        ui.button("Start training", on_click=lambda: svgcontent.start_timer()).style("font-size: 10px")
                        ui.button("Pause training", on_click=lambda: svgcontent.end_timer()).style("font-size: 10px")
                        with ui.row():
                            ui.label("Timer speed:").style("font-size: 10px;")
                            timer_speed_label = ui.label("1").style("font-size: 10px;").bind_text_from(svgcontent, 'timer_speed', lambda x : round((x if x != None else 0.5), 2))
                        timer_speed = ui.slider(min=0.1, max=10, step=0.1).style("font-size:10px; background-color:lightgrey;")
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
                ann.create_model_graphic()
            

            def handle_lr_change(value, ann):
                value = float(value)
                if (value) == 0:
                    value = 0.0001
                print("Updating learning rate", value)
                ann.optimizer = torch.optim.Adam(ann.model.parameters(), lr=value)
                
            with ui.column().style("width: 10vw;"):
                ui.input("Learning rate: ", value=0.001, on_change=lambda e : handle_lr_change(e.value, svgcontent)).style("font-size: 10px;")
                ui.input("Hidden layer 1 size: ", value=4,  on_change=lambda e: handle_layer_1_size_change(e.value, svgcontent)).style("font-size: 10px;")
                ui.input("Hidden layer 2 size: ", value=4,  on_change=lambda e: handle_layer_2_size_change(e.value, svgcontent)).style("font-size: 10px;")
            with ui.column():
                ## show the modele here
                with ui.column():
                    ui.label().bind_text_from(svgcontent, 'stage')
                with ui.column():
                    model_image = ui.interactive_image("/static/annsvg2.svg").style("width: 400px; height: 400px;")
                    model_image.bind_content_from(svgcontent, 'model_graph_image')

        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Model").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            with ui.column():
                ui.label(f"Loss: ").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            with ui.column():
                loss =ui.label("").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
                loss.bind_text_from(svgcontent, 'loss')
    stuff()