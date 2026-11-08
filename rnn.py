"""
Implementing Recurrent Neural Network For AutoRegressive
Even Number Prediction.

Training Data:
    x = [0, 2, 4, 6, ...., 998]
    y = [2, 4, 6, 8, ...., 1000]

Example:
    n          = 5
    start_from = 512
    output     = tensor([514., 516., 518., 520., 522.])
"""

import torch
from torch import nn

class RNN(nn.Module):
    def __init__(
        self, infeatures: int, outfeature: int, hidden_size: int = 32, layer_n: int = 2
    ):
        super().__init__()
        self.rnn = nn.LSTM(infeatures, hidden_size, layer_n, batch_first=True)
        self.fc = nn.Linear(hidden_size, outfeature)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

SEQ_LEN = 10
INFEATURE = 50
OUTFEATURE = 50
# B L IN
x = torch.arange(0, 1000, 2).unsqueeze(dim=0).view(1, SEQ_LEN, INFEATURE)
y = torch.arange(2, 1002, 2).unsqueeze(dim=0).view(1, SEQ_LEN, OUTFEATURE)

def _normalize(x, y):
    # min-max normalization
    mnx, mny = x[:, :, 0].unsqueeze(-1), y[:, :, 0].unsqueeze(-1)
    mx, mxy = x[:, :, -1].unsqueeze(-1), y[:, :, -1].unsqueeze(-1)
    x = (x - mnx) / (mx - mnx)
    y = (y - mny) / (mxy - mny)
    return x, y

x, y = _normalize(x, y)
model = RNN(INFEATURE, OUTFEATURE)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()
epochs = 500
for i in range(epochs):
    optim.zero_grad()
    y_pred = model(x)
    loss = mse_loss(y_pred, y)
    if i % 10 == 0:
        print(f"Loss : {loss}")
    loss.backward()
    optim.step()
x_test = torch.arange(512, 612, 2).float().unsqueeze(dim=0).view(1, 1, 50)
mn = x_test[:, :, 0].unsqueeze(-1)
mx = x_test[:, :, -1].unsqueeze(-1)
x_test_norm = (x_test - mn) / (mx - mn)
with torch.no_grad():
    y_test = model(x_test_norm)
    y_real = y_test * (mx - mn) + (mn + 2)
    print(torch.round(y_real[0, 0, :5]))
