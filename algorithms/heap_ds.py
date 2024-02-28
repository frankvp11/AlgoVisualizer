
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
        self.heapify(0)
        return max_val

    def delete(self, i):
        self.arr[i] = float("inf")
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
        queue = [root]
        i = 1

        while i < len(self.arr):
            current = queue.pop(0)

            left_val = self.arr[i]
            if left_val is not None:
                current.left = Tree(left_val)
                queue.append(current.left)

            i += 1

            if i < len(self.arr):
                right_val = self.arr[i]
                if right_val is not None:
                    current.right = Tree(right_val)
                    queue.append(current.right)

                i += 1

        return root