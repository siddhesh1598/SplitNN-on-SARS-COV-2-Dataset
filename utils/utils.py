# import
import matplotlib.pyplot as plt

# initializing the required arguments
class Arguments():
    
    def __init__(self):
        self.batch_size = 128
        self.test_batch_size = 1000
        self.epochs = 50
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 2
        self.save_model = False
        self.split = 0.8
        self.image_size = (256, 256)
        self.input_size = 65536
        self.output_size = 2


# plot images from the data loader
def plotImages(datasetLoader):
    examples = enumerate(datasetLoader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(3):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Label: {}".format(
            "covid" if example_targets[i] else "non-covid"))        
        plt.xticks([])
        plt.yticks([])
        
    fig.show()


# plot train and test loss
def plotLoss(train_counter, train_losses, 
        test_counter, test_losses, path):
    fig = plt.figure()

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    
    fig.savefig(path)
    fig.show()




