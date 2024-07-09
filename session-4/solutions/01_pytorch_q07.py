# My choice, a tw layer neural network with RELU activation:
# - Linear layer with 10 neurons
# - Relu activation function
# - Linear layer with 5 neurons
# - Relu activation function
# - Linear layer with 2 neurons
# - LogSoftmax output

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.LogSoftmax(dim=1)
)
