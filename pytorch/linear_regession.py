import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


x = [1.1, 2.1, 4.3, -1.2, -2.4, -3.5]
y = [0.7, 1.0, 3.2, -1.1, -2.1, -3.4]

print("calcuate a, b for ax+b=y,")
print("    x is", x)
print("    y is", y)
print("")

# train data
x = torch.Tensor(x).view(6, 1)
y = torch.Tensor(y).view(6, 1)

# forward => AX + B
forward = Net()

learning_rate = 0.01

optimizer = optim.SGD(forward.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for i in range(0, 1000):
    # without optimizer:
    # -> forward.zero_grad()
    optimizer.zero_grad()

    output = forward(x)

    loss = criterion(output, y)
    loss.backward()

    # without optimizer:
    # -> for f in forward.parameters():
    #        f.data.sub_(f.grad.data * learning_rate)
    optimizer.step()

params = list(forward.parameters())
print("a = %.6f, b = %.6f" % (params[0].item(), params[1].item()))

