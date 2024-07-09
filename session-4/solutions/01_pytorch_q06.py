class My_network(nn.Module):
    def __init__(self, d_in, d_h1, d_out):
        super().__init__()
        # Create the parameters of the network
        W_hidden = torch.empty(d_in, d_h1, dtype=torch.float)
        b_hidden = torch.empty(d_h1, dtype=torch.float)

        W_output = torch.empty(d_h1, d_out, dtype=torch.float)
        b_output = torch.empty(d_out, dtype=torch.float)

        # Initialize the parameters with nn.init.kaiming_uniform_
        # and nn.init.normal_.
        # One could have chosen another type of initialization
        nn.init.kaiming_uniform_(W_hidden)
        nn.init.normal_(b_hidden)

        nn.init.kaiming_uniform_(W_output)
        nn.init.normal_(b_output)

        # Make tensors learnable parameters with torch.nn.Parameter
        self.W_hidden = torch.nn.Parameter(W_hidden)
        self.b_hidden = torch.nn.Parameter(b_hidden)

        self.W_output = torch.nn.Parameter(W_output)
        self.b_output = torch.nn.Parameter(b_output)

    def forward(self, x):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, d_in)
        """
        # Compute the forward pass
        h = torch.matmul(x, self.W_hidden) + self.b_hidden
        h = torch.sigmoid(h)
        h = torch.matmul(h, self.W_output) + self.b_output
        h = F.softmax(h, dim=1)

        return h
