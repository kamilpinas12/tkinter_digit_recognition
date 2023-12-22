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


    def add_coordinates(self, x, y):
        if 1 <= x <= self.window_size[0]-2 and 1 <= y <= self.window_size[1]-2:
            self.x_cord = np.append(self.x_cord, x)
            self.y_cord = np.append(self.y_cord, y)
                    

    def clear(self):
        self.x_cord = np.array([]).astype(int)
        self.y_cord = np.array([]).astype(int)


    def process_image(self):
        # function takes coordinates of drawn points and convert it to list of 28x28 bitmaps of numbers which can be process by neural network
        # sometimes when you draw one digit this funciton convert it to several digit, it happens when you draw to fast and tkinter gives to little coordinates 
        # thats why there is multiple_digit parameter, when you use this function with add_data.py there are always one digit so you don't want to split, hope it make sense
        # this function isn't the best honestly, but works in most cases
        if self.x_cord.size == 0 or self.x_cord.size == None:
            return None


        coordinates = list(zip(self.x_cord, self.y_cord))
        digit_number = 0
        digit_cord = [[]]

        # separate digits
        coordinates.sort(key=lambda x: x[0])
        digit_number = 0
        digit_cord[digit_number].append(coordinates[0])

        for i in range(1, len(coordinates)):
            if coordinates[i][0] - coordinates[i-1][0] < 20:
                digit_cord[digit_number].append(coordinates[i])
            else:
                digit_cord.append([])
                digit_number += 1
                digit_cord[digit_number].append(coordinates[i])
                        




        images = []
        for i in range(digit_number + 1):

            if len(digit_cord[i]) < 2:
                continue

            x_cord = [x[0] for x in digit_cord[i]]
            y_cord = [y[1] for y in digit_cord[i]]
            img = np.zeros((self.size_x, self.size_y))


            max_y = max(y_cord)
            min_x = min(x_cord)
            min_y = min(y_cord)
            max_x = max(x_cord)
            height = max_y - min_y
            width = max_x - min_x

            #crop image
            if height > width:
                size = height
                x_cord -= min_x - int((height - width)/2)       
                y_cord -= min_y      
            else:
                size = width
                y_cord -= min_y - int((width - height)/2) 
                x_cord -= min_x



            for x, y in list(zip(x_cord, y_cord)):
                x_1 = int(17*x/size) + 6
                y_1 = int(17*y/size) + 6
                for i in range(-1, 1):
                    for j in range(-1, 1):
                        try:
                            if j == 0:
                                img[y_1 + i, x_1 + j] = 1
                            elif img[y_1 + i, x_1 + j] == 0:
                                img[y_1 + i, x_1 + j] = 0.5
                        except:
                            continue          
            
            images.append(img.reshape((784, 1)))

        return images
    


    def append_data_to_csv(self, img, label):
        img = img.reshape(-1)
        img = np.insert(img, 0, label)
        img = img.reshape(1,img.size)
        df = pd.DataFrame(img)
        df.to_csv('data/data_3.csv', mode='a', index=False, header=False)




