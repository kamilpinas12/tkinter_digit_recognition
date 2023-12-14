import numpy as np
import pandas as pd

class Image():
    def __init__(self, window_size):
        self.x_cord = np.array([]).astype(int)
        self.y_cord = np.array([]).astype(int)
        self.size_x = 28
        self.size_y = 28
        self.window_size = window_size
        self.bitmap = np.zeros((self.size_x, self.size_y))
        self.new_data = None


    def add_coordinates(self, x, y):
        if 1 <= x <= self.window_size[0]-2 and 1 <= y <= self.window_size[1]-2:
            self.x_cord = np.append(self.x_cord, x)
            self.y_cord = np.append(self.y_cord, y)
                    



    def clear(self):
        self.x_cord = np.array([])
        self.y_cord = np.array([])
        self.bitmap = np.zeros((self.size_x, self.size_y))



    def process_image(self):
        if self.x_cord.size == 0 or self.x_cord.size == None:
            return None
        
        # add thicker lines
        for x, y in list(zip(self.x_cord, self.y_cord)):
            for i in range(0):
                for j in range(0):
                    try:
                        self.x_cord = np.append(self.x_cord, x + i)
                        self.y_cord = np.append(self.y_cord, y + j)
                    except:
                        continue

        max_x = np.max(self.x_cord)
        max_y = np.max(self.y_cord)
        min_x = np.min(self.x_cord)
        min_y = np.min(self.y_cord)
        height = max_y - min_y
        width = max_x - min_x

        #crop image
        if height > width:
            size = height
            self.x_cord -= min_x - int((height - width)/2)       
            self.y_cord -= min_y      
        else:
            size = width
            self.y_cord -= min_y - int((width - height)/2) 
            self.x_cord -= min_x


        for x, y in list(zip(self.x_cord, self.y_cord)):
            x_1 = int(24*x/size) + 2
            y_1 = int(24*y/size) + 2
            for i in range(-1, 1):
                for j in range(-1, 1):
                    try:
                        self.bitmap[abs(y_1 + i), abs(x_1 + j)] = 1
                    except:
                        continue
            self.bitmap[y_1, x_1] = 1

        self.bitmap_to_csv()



    def get_converted_image(self):
        return self.bitmap.reshape((784, 1))



    def append_data_to_csv(self, label):
        img = self.bitmap.reshape(-1)
        img = np.insert(img, 0, label)
        img = img.reshape(1,img.size)
        df = pd.DataFrame(img)
        df.to_csv('data_2.csv', mode='a', index=False, header=False)





    def bitmap_to_csv(self):
        df = pd.DataFrame(self.bitmap)
        for i in range(int(df.shape[0])):
            for j in range(int(df.shape[1])):
                if df.iloc[i, j] == 0:
                    df.iloc[i, j] = ' '
                else:
                    df.iloc[i, j] = '@'

        df.to_csv('bitmap.csv', index=False, header=False)


