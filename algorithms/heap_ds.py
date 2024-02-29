
import sys

sys.path.append('/home/frank/Projects/Python/Algorithms')

from ModelMaker.graphicsSVG2.Circle import Circle
from ModelMaker.graphicsSVG2.Line import Line
from ModelMaker.graphicsSVG2.Text import Text
from ModelMaker.graphicsSVG2.Arrow  import Arrow
from ModelMaker.graphicsSVG2.ShapeCollection import ShapeCollection
from .tree import Tree




class Heap:
    def __init__(self, arr):
        self.arr = arr
        self.size = len(arr)
        self.polygons = ShapeCollection()
        self.polygons.clear_all_polygons()
        self.content = ""
        self.current_x=  500
        self.current_y = 100

    def heapify(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < self.size and self.arr[left] > self.arr[largest]:
            largest = left
        if right < self.size and self.arr[right] > self.arr[largest]:
            largest = right
        if largest != i:
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            self.heapify(largest)


    def build_heap(self):
        for i in range(self.size // 2 - 1, -1, -1):
            self.heapify(i)

    def insert(self, val):
        self.arr.append(val)
        self.size += 1
        i = self.size - 1
        while i > 0 and self.arr[i] > self.arr[(i - 1) // 2]:
            self.arr[i], self.arr[(i - 1) // 2] = self.arr[(i - 1) // 2], self.arr[i]
            i = (i - 1) // 2

    def extract_max(self):
        if self.size < 1:
            return
        max_val = self.arr[0]
        self.arr[0] = self.arr[self.size - 1]
        self.size -= 1
        self.arr.pop()
        self.heapify(0)
        return max_val

    def delete(self, i):
        self.arr[i] = float("-inf")  # Change to negative infinity
        while i > 0 and self.arr[i] > self.arr[(i - 1) // 2]:
            self.arr[i], self.arr[(i - 1) // 2] = self.arr[(i - 1) // 2], self.arr[i]
            i = (i - 1) // 2
        self.extract_max()


    def get_max(self):
        if self.size < 1:
            return
        return self.arr[0]

    def heap_sort(self):
        for i in range(self.size - 1, 0, -1):
            self.arr[0], self.arr[i] = self.arr[i], self.arr[0]
            self.size -= 1
            self.heapify(0)
        return self.arr


    def heap_to_binary_tree(self):
        if not self.arr:
            return None

        root = Tree(self.arr[0])
        self.polygons.clear_all_polygons()
        queue = [(root, 750, 200)]
        circle = Circle(750, 200, 45, color='darkgray', transparency=0.9)
        circle.give_outline(color='black', thickness=2)
        text = Text(self.arr[0], 747, 200, font_size=40)
        self.polygons.add_polygon(circle)
        self.polygons.add_polygon(text)
        i = 1

        while i < len(self.arr):
            current, x, y = queue.pop(0)

            left_val = self.arr[i]
            if left_val is not None:
                circle = Circle(x - 250, y + 250, 45, color='darkgray', transparency=0.9)
                circle.give_outline(color='black', thickness=2)
                text = Text(left_val, x - 255, y + 250, font_size=40)
                self.polygons.add_polygon(circle)
                self.polygons.add_polygon(text)

                current.left = Tree(left_val)
                new_x = 0.875 * ( (x-250) - x) + x
                new_y = 0.875 * ( (y+250) - y) + y
                start_x  = 0.875 * (x - (x-250)) + (x-250)
                start_y  = 0.875 * (y - (y+250)) + (y+250)
                arrow = Arrow(start_x, start_y, new_x, new_y, head_width=20)
                
                self.polygons.polygons.extend(arrow.polygons)

                queue.append((current.left, x - 250, y + 250))

            i += 1

            if i < len(self.arr):
                right_val = self.arr[i]
                if right_val is not None:
                    circle = Circle(x + 250, y + 250, 45, color='darkgray', transparency=0.9)
                    circle.give_outline(color='black', thickness=2)
                    text = Text(right_val, x + 245, y + 250, font_size=40)
                    self.polygons.add_polygon(circle)
                    self.polygons.add_polygon(text)

                    current.right = Tree(right_val)
                    new_x = 0.875 * ( (x+250) - x) + x
                    new_y = 0.875 * ( (y+250) - y) + y
                    start_x  = 0.875 * (x - (x+250)) + (x+250)
                    start_y  = 0.875 * (y - (y+250)) + (y+250)
                    arrow = Arrow(start_x, start_y, new_x, new_y, head_width=20)
                    self.polygons.polygons.extend(arrow.polygons)

                    queue.append((current.right, x + 250, y + 250))

                i += 1

        self.content = self.polygons.to_svg()

        return root