import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                    
    def unprocessed_data_to_csv(self):
        pass


    def clear(self):
        self.x_cord = np.array([])
        self.y_cord = np.array([])
        self.bitmap = np.zeros((self.size_x, self.size_y))



    def process_image(self):
        if self.x_cord.size == 0 or self.x_cord.size == None:
            return None
        
        coordinates = list(zip(self.x_cord, self.y_cord))

        # add thicker lines
        # for x, y in coordinates:
        #     for i in range(0):
        #         for j in range(0):
        #             try:
        #                 self.x_cord = np.append(self.x_cord, x + i)
        #                 self.y_cord = np.append(self.y_cord, y + j)
        #             except:
        #                 continue


        # # separate digits
        # digit_cord = []
        # print(type(coordinates))
        # coordinates.sort(key=lambda x: x[0])
        # digit_number = 0

        # #podzielenie współrzędnych na 
        # for i in range(1, len(coordinates)):
        #     if coordinates[i-1][0] - coordinates[i][0] < 20:
        #         digit_cord[digit_number].append(coordinates)
        #     else:
        #         digit_cord.append([])
        #         digit_number += 1
        # print(digit_number)
                    

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
            x_1 = int(17*x/size) + 6
            y_1 = int(17*y/size) + 6
            for i in range(-1, 1):
                for j in range(-1, 1):
                    try:
                        if j == 0:
                            self.bitmap[y_1 + i, x_1 + j] = 1
                        elif self.bitmap[y_1 + i, x_1 + j] == 0:
                            self.bitmap[y_1 + i, x_1 + j] = 0.5
                    except:
                        continue

        self.bitmap_to_csv()


    def show_image(self):
        plt.clf()
        plt.gray()
        plt.imshow(self.bitmap, interpolation='nearest')
        plt.show()



    def get_converted_image(self):
        return self.bitmap.reshape((784, 1))



    def append_data_to_csv(self, label):
        img = self.bitmap.reshape(-1)
        img = np.insert(img, 0, label)
        img = img.reshape(1,img.size)
        df = pd.DataFrame(img)
        df.to_csv('data/data_3.csv', mode='a', index=False, header=False)





    def bitmap_to_csv(self):
        df = pd.DataFrame(self.bitmap)
        for i in range(int(df.shape[0])):
            for j in range(int(df.shape[1])):
                if df.iloc[i, j] == 0:
                    df.iloc[i, j] = ' '
                else:
                    df.iloc[i, j] = '@'

        df.to_csv('data/bitmap.csv', index=False, header=False)


