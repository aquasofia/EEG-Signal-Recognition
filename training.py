from CNN import CNNSystem
from torch import Tensor
from dataset import EEG
from data_init import *
import json
import time
import os


def main():

    directory = 'data/train'

    file = os.listdir(directory)[0]

    # Load training data from the directory
    dataset_training = EEG(data_dir='data/train/',
                           file=file)

    # Load the model and training parameters from external file
    # configs = json.load(open('configs.json', 'r'))

    # Obtain a data loader for training set
    loader_training = get_data_loader(
        dataset=dataset_training,
        batch_size=1,
        shuffle=True)

    # Create an instance of the CNN model
    cnn = CNNSystem()
    cnn = cnn.to(torch.double)
    # For a binary classifier ADAM optimizer
    optimizer = torch.optim.Adam(cnn.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    # Initialize arrays for losses
    losses_training = []
    # Measure performance for 100 epochs
    tic = time.perf_counter()
    num_correct = 0
    num_samples = 0

    # Set training loop to max epoch
    for epoch in range(50):
        # 1. Training the neural network with training set
        print('-----------------------------')
        print(' Running model in training set')

        for x, y in loader_training:
            # Reset gradients from the previous round
            optimizer.zero_grad()
            # Feed data to the model
            y_hat = cnn(x)
            # Calculate loss and append to training losses array
            y = torch.LongTensor(y)
            loss_training = loss_func(y_hat, y)
            losses_training.append(loss_training.item())
            # print(' loss', loss_training.item())
            # Initiate backpropagation on the basis of the loss
            loss_training.backward()
            # Optimize network weights
            optimizer.step()
            _, predictions = y_hat.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            print(f' With accuracy of {float(num_correct) / float(num_samples) * 100:.2f}')
    cnn.eval()

    toc = time.perf_counter()

    print('\n', 'RESULTS')
    print('-----------------------------')
    print(f" Model training time {toc - tic:0.4f} seconds")
    print(' Training loss: ', Tensor(losses_training).mean().item())
    print(f' Classified in total of {num_correct}/{num_samples} samples')
    print(f' With accuracy of {float(num_correct) / float(num_samples) * 100:.2f}')
    print('-----------------------------')
    print('\n', 'EPOCH ', epoch, '| LOSS MEAN ', Tensor(losses_training).mean().item())


    torch.save(cnn, 'cnn_model')


if __name__ == '__main__':
    main()
