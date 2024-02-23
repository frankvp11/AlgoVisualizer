
from nicegui import ui
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from ModelMaker.graphicsSVG2.CustomPolygon import CustomPolygon

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
    def __init__(self):
        self.X, self.y = self.get_points()
        self.model = SimpleModel(2, 4, 4, 3)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = 0
        self.graph = self.generate_graph()
        self.graph.move_all(100, 100)
        self.content = self.graph.to_svg()
        self.previous_contours = []
        self.current_epoch = 0
        self.timer_speed = 1
        self.timer = None

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
            self.timer_speed = 1
        self.timer = ui.timer(interval=(1 / self.timer_speed), callback=self.run_epoch)

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
        self.current_epoch += 1
        self.optimizer.zero_grad()
        y_pred = self.model(self.X)
        self.loss = self.loss_function(y_pred, self.y)
        self.loss.backward()
        self.optimizer.step()
        self.loss = self.loss.item()
        self.update_boundaries()
        return self.loss


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

def run_epoch(model, loss_function, optimizer, X, y):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def add():
    with ui.header():
        with ui.link(target="/"):
            ui.button(icon="home")


    @ui.refreshable
    def stuff():
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Artificial Neural Networks").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
        with ui.row():
            with ui.column().style("width: 40vw;"):
                image = ui.interactive_image(source="/static/annsvg.svg")
                svgcontent= ANN()
                
                
                image.bind_content_from(svgcontent, 'content')
            with ui.column().style("width: 40vw; "):
                with ui.row():
                    with ui.column():
                        ui.button("Generate new graph", on_click=lambda: stuff.refresh()).style("font-size: 10px")
                    with ui.column():
                        ui.button("Start training", on_click=lambda: svgcontent.start_timer()).style("font-size: 10px")
                        ui.button("Pause training", on_click=lambda: svgcontent.stop_timer()).style("font-size: 10px")
                        with ui.row():
                            ui.label("Timer speed:").style("font-size: 10px;")
                            timer_speed_label = ui.label("1").style("font-size: 10px;").bind_text_from(svgcontent, 'timer_speed', lambda x : round(1/(x if x != None else 1), 2))
                        timer_speed = ui.slider(min=1, max=10, step=1).style("font-size:10px; background-color:lightgrey;")
                        timer_speed.bind_value_to(svgcontent, 'timer_speed')
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Model").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            with ui.column():
                ui.label(f"Loss: ").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            with ui.column():
                loss =ui.label("").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
                loss.bind_text_from(svgcontent, 'loss')
    stuff()