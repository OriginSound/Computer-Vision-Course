import os
import struct
import numpy as np
import pdb

path = "Computer-Vision-Course/homework1/MNIST/raw/"

class Dataset:
    def __init__(self, train=True):
        if train:
            img_file, label_file = "train-images.idx3-ubyte", "train-labels.idx1-ubyte"
        else:
            img_file, label_file = "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"

        # read the images 
        with open(os.path.join(path, img_file), 'rb') as img_path:
            _, images_num, rows, cols = struct.unpack('>IIII', img_path.read(16))
            self.images = np.fromfile(img_path, dtype=np.uint8).reshape(images_num, rows * cols)
        self.images = self.images / 255.0 

        # read the labels
        with open(os.path.join(path, label_file), 'rb') as label_path:
            _, labels_num = struct.unpack('>II', label_path.read(8))
            self.labels = np.fromfile(label_path, dtype=np.uint8)
        
        assert images_num == labels_num, "Inconsistent lengths of data and labels!"
        self.length = images_num
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.length
    

class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset 
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        idx, length = 0, len(self.dataset)
        indices = np.random.permutation(length)

        while idx*self.batch_size < length:
            batch_indices = indices[idx*self.batch_size : (idx+1)*self.batch_size]
            idx += 1
            yield self.dataset[batch_indices]


if __name__ == "__main__":
    dataset = Dataset(train=False)
    import matplotlib.pyplot as plt 
    image, label = dataset[10]
    print(label)
    plt.imshow(image.reshape(28, 28))
    print(image.reshape(28,28)[10:18, 10:18])
    plt.show()
