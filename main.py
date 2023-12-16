import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from image_processing import Image
from nn import NN
import numpy as np


class PaintApp:
    def __init__(self, root, x = None, *args):
        self.root = root
        self.canvas_width = 400
        self.canvas_height = 400
        self.pixel_vector = x


        self.predicted_label = tk.StringVar()



        
        self.image = Image((self.canvas_width, self.canvas_height))
        self.nn = NN()

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", bd=3, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_navbar()
        self.setup_tools()
        self.setup_events()
        self.prev_x = None
        self.prev_y = None



    def setup_navbar(self):
        self.navbar = tk.Menu(self.root)
        self.root.config(menu=self.navbar)

        # File menu
        self.file_menu = tk.Menu(self.navbar, tearoff=False)
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Clear bitmap", command=self.take_snapshot)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)


    def show_prediction(self, A):
        s = str(np.argmax(A)) + " ~ " + str(int(np.max(A)*100)) + " %"
        self.predicted_label.set(s)


    def setup_tools(self):
        self.selected_tool = "pen"
        self.selected_color = "black"
        self.selected_size = 6
        self.selected_pen_type = "line"

        self.tool_frame = ttk.LabelFrame(self.root, text="Tools")
        self.tool_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)


        self.clear_button = ttk.Button(self.tool_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.TOP, padx=5, pady=5)

        self.clear_button = ttk.Button(self.tool_frame, text="Predict", command=self.predict)
        self.clear_button.pack(side=tk.TOP, padx=5, pady=5)

        self.show_image_button = ttk.Button(self.tool_frame, text="Show img", command=self.show_img)
        self.show_image_button.pack(side=tk.TOP, padx=5, pady=5)


        self.txt_label = ttk.Label(self.tool_frame, text="Prediction: ")
        #self.txt_label.pack(side=tk.TOP, padx=5, pady=20)
        self.txt_label.place(y=200, x=13)


        self.prediction_label = ttk.Label(self.tool_frame, textvariable=self.predicted_label)
        self.prediction_label.place(y = 222, x = 15)



    def show_img(self):
        self.image.show_image()



    def setup_events(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release)



    def predict(self):
        self.image.process_image()
        img = self.image.get_converted_image()
        res = self.nn.forward_prop(img).reshape(-1)

        x = np.linspace(0,9,10)
        self.show_prediction(res)



    def draw(self, event):
        if self.selected_tool == "pen":
            if self.prev_x is not None and self.prev_y is not None:
                if self.selected_pen_type == "line":
                    self.image.add_coordinates(event.x, event.y)
                    self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y, fill=self.selected_color,
                                            width=self.selected_size, smooth=True)
            self.prev_x = event.x
            self.prev_y = event.y



    def release(self, event):
        self.prev_x = None
        self.prev_y = None



    def clear_canvas(self):
        self.image.process_image()
        self.image.clear()
        self.canvas.delete("all")

    def take_snapshot(self):
        pass

    def undo(self):
        items = self.canvas.find_all()
        if items:
            self.canvas.delete(items[-1])
    
    def get_pixel_vector(self):
        return self.pixel_vector

    

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Paint Application")
    app = PaintApp(root)
    root.mainloop()


