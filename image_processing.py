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
        

        # add thicker lines
        # for x, y in coordinates:
        #     for i in range(0):
        #         for j in range(0):
        #             try:
        #                 self.x_cord = np.append(self.x_cord, x + i)
        #                 self.y_cord = np.append(self.y_cord, y + j)
        #             except:
        #                 continue


       
        # separate digits
        coordinates = list(zip(self.x_cord, self.y_cord))
        digit_cord = [[]]
        coordinates.sort(key=lambda x: x[0])
        digit_number = 0
        digit_cord[digit_number].append(coordinates[0])

        for i in range(1, len(coordinates)):
            if coordinates[i][0] - coordinates[i-1][0] < 15:
                digit_cord[digit_number].append(coordinates[i])
            else:
                digit_cord.append([])
                digit_number += 1
                    
        
        images = []


        for i in range(digit_number + 1):
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
            self.bitmap_to_csv(img)            
            
            images.append(img.reshape((784, 1)))

        return images

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





    def bitmap_to_csv(self, img):
        df = pd.DataFrame(img)
        for i in range(int(df.shape[0])):
            for j in range(int(df.shape[1])):
                if df.iloc[i, j] == 0:
                    df.iloc[i, j] = ' '
                else:
                    df.iloc[i, j] = '@'

        df.to_csv('data/bitmap.csv', index=False, header=False)


