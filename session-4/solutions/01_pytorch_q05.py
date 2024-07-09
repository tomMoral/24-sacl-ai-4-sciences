net = Network()
print(net.hidden.weight.shape)
print(net.hidden.bias.shape)
print(net.output.weight.shape)
print(net.output.bias.shape)

# Or simply
for param in net.parameters():
    print(param.shape)