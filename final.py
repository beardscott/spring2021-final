# -*- coding: utf-8 -*-
"""
Data import code from
https://towardsdatascience.com/python-how-to-get-live-market-data-less-than-0-1-second-lag-c85ee280ed93

Code from
https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632
was utilized to create the algorithm

"""

# Raw Package
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

#Data Source
import yfinance as yf

#Data viz
# import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.renderers.default='browser' #Firefox renderer
# pio.renderers.default='svg' #Spyder renderer

#Interval required 1 minute
tickers = ['GME']
period = '1wk'
interval = '1d'
start = '2020-01-01'
end = '2021-02-05'
data = yf.download(tickers=tickers, start=start, end=end, interval=interval)

input_dim = 1
hidden_dim = 16
num_layers = 2
output_dim = 1
num_epochs = 100

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def split_data(stock, lookback):
    """ 
    """
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

lookback = 20 # choose sequence length
x_train, y_train, x_test, y_test = split_data(price, lookback)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_gru)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
y_train_pred = y_train_pred.detach().numpy()
y_test_pred = model(x_test).detach().numpy()

y_train_out = scaler.inverse_transform(y_train_pred)
y_test_out = scaler.inverse_transform(y_test_pred)

fig0 = plt.figure(0)
plt.plot(range(len(data[['Close']])),data[['Close']].values)
plt.plot(range(lookback,len(data[['Close']])-len(y_test_pred)),y_train_out)
plt.plot(range(len(data[['Close']])-len(y_test_pred),len(data[['Close']])),y_test_out)
plt.legend(['Price','Training','Testing'])
plt.xlabel('Days since 01/01/2021')
plt.ylabel('MSFT Price (USD)')
plt.title('Model with 20 day lookback period')

fig1 = plt.figure(1)
plt.plot(hist)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training with 20 day lookback period')
