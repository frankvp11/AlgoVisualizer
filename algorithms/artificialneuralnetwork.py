
from nicegui import ui
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from ModelMaker.graphicsSVG2.CustomPolygon import CustomPolygon

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
    
    def run_epoch(self):
        self.optimizer.zero_grad()
        y_pred = self.model(self.X)
        self.loss = self.loss_function(y_pred, self.y)
        self.loss.backward()
        self.optimizer.step()
        self.loss = self.loss.item()
        return self.loss


    def generate_graph(self):
        points = []
        color_mapping = {0: "red", 1: "blue", 2: "green"}
        for (x, x2), color in zip(self.X, self.y):
            points.append(Circle(x.item()*10, x2.item()*10, 1, color=color_mapping[color.item()])) 
        return ShapeCollection(points)
    def update_boundaries(self):
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = self.model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        decision_boundary_polygon = CustomPolygon(vertices=np.c_[xx.ravel(), yy.ravel()], color='blue')


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
        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Data").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            image = ui.interactive_image(source="/static/annsvg.svg")
            svgcontent= ANN()
            
            
            image.bind_content_from(svgcontent, 'content')

        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.button("Generate new graph", on_click=lambda: stuff.refresh())
            ui.button("Train one epoch", on_click=lambda: svgcontent.run_epoch())

        with ui.row().style("width: 100vw; justify-content:center; text-align:center; align-items:center;"):
            ui.label("Model").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            with ui.column():
                ui.label(f"Loss: ").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
            with ui.column():
                loss =ui.label("").style("font-size: 20px; font-weight: bold; margin-bottom: 20px; justify-content:center;")
                loss.bind_text_from(svgcontent, 'loss')
    stuff()