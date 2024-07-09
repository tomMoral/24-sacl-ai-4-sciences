
class ResNetBlock(nn.Module):
    def __init__(self, d, dropout_rate):
        super().__init__()
        self.d = d
        self.dropout_rate = dropout_rate
        self.norm = nn.BatchNorm1d(d)
        self.linear0 = nn.Linear(d, d)
        self.linear1 = nn.Linear(d, d)

    def forward(self, x):
        z = x
        z = self.norm(z)
        z = self.linear0(z)
        z = F.relu(z)
        z = F.dropout(z, self.dropout_rate)
        z = self.linear1(z)
        z = F.dropout(z, self.dropout_rate)
        x = x + z
        return x

class ResNet(nn.Module):
    def __init__(self, d, d_out, dropout_rate, n_resnet_blocks):
        super().__init__()
        self.d = d
        self.d_out = d_out
        self.dropout_rate = dropout_rate
        self.n_resnet_blocks = n_resnet_blocks
        self.first_linear_layer = nn.Linear(d, d)
        self.resnetblocks = nn.Sequential(
           *(ResNetBlock(d, dropout_rate) for _ in range(n_resnet_blocks))
        )
        self.last_normalization = nn.BatchNorm1d(d)
        self.last_linear_layer = nn.Linear(d, d_out)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.first_linear_layer(x)
        x = self.resnetblocks(x)
        x = self.last_normalization(x)
        x = nn.ReLU()(x)
        x = self.last_linear_layer(x)
        x = self.softmax(x)
        return x
