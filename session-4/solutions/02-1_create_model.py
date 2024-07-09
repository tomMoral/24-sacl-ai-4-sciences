# create the model
import torch


class BachSynth(torch.nn.Module):

    def __init__(self):
        ###################
        # TODO

        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=n_x, hidden_size=256, batch_first=True
        )
        self.lin_1 = torch.nn.Linear(256, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.lin_2 = torch.nn.Linear(256, n_x)

        # END TODO
        ###################

    def forward(self, X):
        ###################
        # TODO

        h_x, c_x = torch.zeros((2, 1, X.shape[0], 256))
        out, h_x = self.lstm(X, (h_x, c_x))
        out = self.lin_1(out)
        out = self.dropout(out)
        return self.lin_2(out)

        # END TODO
        ###################


model = BachSynth()
print(model)
