#import
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imutils import paths


class DataGenerator(Dataset):
    """
    Dataset generator for PyTorch models

    Args:
        images = list of paths to all images
        labels = list of labels corresponding to 
                        all the images
        transform = transformations to be applied to 
                    the images
	"""

    # initialize images, labels and transforms
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    # return length of images
    def __len__(self):
        return len(self.images)

    # return transformed images with their labels
    def __getitem__(self, idx):
        image = self.images[idx]
        labels = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return [image, labels]


class DatasetLoader:
    """
    Train and Test Dataset Loader for PyTorch models

    Args:
        data_dir = dictionary with labels as keys and
                    images directories as values
        args = required arguments
	"""

    # initialize train and test images with their labels 
    def __init__(self, data_dir, args):
        self.args = args
        self.trainImages = {}
        self.testImages = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
            
        for (label, images_dir) in data_dir.items():
            all_images_list = list(paths.list_images(images_dir))
            split = int(len(all_images_list) * self.args.split)
            
            self.trainImages[label] = all_images_list[:split]
            self.testImages[label] = all_images_list[split:]
        

    def generateTrainDataset(self):
        """
        Returns:
                trainDataLoader = data loader for training data
        """
        
        images = list()
        labels = list()

        # resize the images and masks
        for (label, images_dir) in self.trainImages.items():
            for imagePath in images_dir:
                image = cv2.resize(cv2.imread(imagePath, 0), self.args.image_size)
                images.append(image)
                labels.append(1 if label == "covid" else 0)

        # generate the dataset loader for the image and mask dataset
        print("[INFO] Generating Training Dataset Loader ...")
        print("[INFO] Total number of images for training dataset: {}".format(
                                len(images)))

        train_set = DataGenerator(images, labels, transform=self.transform)
        trainDataLoader = DataLoader(
            train_set, batch_size=self.args.batch_size, 
            shuffle=True
        )

        return trainDataLoader
    
    def generateTestDataset(self):
        """
        Returns:
                testDataLoader = data loader for testing data
        """

        images = list()
        labels = list()

        # resize the images and masks
        for (label, images_dir) in self.testImages.items():
            for imagePath in images_dir:
                image = cv2.resize(cv2.imread(imagePath, 0), self.args.image_size)
                images.append(image)
                labels.append(1 if label == "covid" else 0)

        # generate the dataset loader for the image and mask dataset
        print("[INFO] Generating Testing Dataset Loader ...")
        print("[INFO] Total number of images for testing dataset: {}".format(
                                len(images)))

        test_set = DataGenerator(images, labels, transform=self.transform)
        testDataLoader = DataLoader(
            test_set, batch_size=self.args.test_batch_size, 
            shuffle=True
        )

        return testDataLoader