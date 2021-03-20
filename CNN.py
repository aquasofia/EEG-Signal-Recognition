from torch import Tensor, tensor
from torch.nn import Module, Conv1d, MaxPool1d, ReLU, Linear, Dropout, Sequential
from EEG_Analysis import dataset


class CNNSystem(Module):

    def __init__(self) \
            -> None:
        super().__init__()

        # First layer : CNN xx neurons

        self.layer1 = Sequential(Conv1d(in_channels=1, out_channels=16, kernel_size=10),
                                 ReLU(),
                                 MaxPool1d(kernel_size=3))

        # 2nd layer : CNN 32 neurons padding c6
        self.layer2 = Sequential(Conv1d(16, 16, kernel_size=15),
                                 ReLU(),
                                 MaxPool1d(kernel_size=5))

        # 3rd layer CNN 16 neurons
        self.layer3 = Sequential(Conv1d(32, 32, kernel_size=20),
                                 ReLU(),
                                 MaxPool1d(kernel_size=5))

        # need to know input size for 1st mlp layers
        # original self.mlp1 = Linear(16, 10)
        self.mlp1 = Linear(208, 20)
        # original self.mlp2 = Linear(10, 5)
        self.mlp2 = Linear(20, 10)

    def forward(self, x: Tensor) \
            -> Tensor:

        # Neural network layer stacking
        x = x[:, :, 0:250]
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        t = self.mlp1(x)
        y_hat = self.mlp2(t)

        return y_hat

# EOF
