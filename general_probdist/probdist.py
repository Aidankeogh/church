from torch import sigmoid
import torch
import random

class GeneralProbdist:
    def __init__(self, probs) -> None:
        self.probs = [torch.Tensor(e) for e in probs]
    def sample(self):
         logit = 0
        for i, prob_arr in enumerate(self.probs):
            temp = torch.Tensor([t / 2**j for j, t in enumerate(outs + [1])])
            logit += torch.dot(temp, prob_arr) / 2**i
            prob = sigmoid(logit)
            c = (random.random() <= prob.item())
            outs.append(c)
        return sum(j<<i for i,j in enumerate(reversed(outs)))
    
    def get_prob(self, value):
        bin_str = bin(value)[2:].zfill(len(self.probs))
        bin_value = [int(val) for val in list(bin_str)]
        cumulative_probability = 1
        logit = 0
        for i, prob_arr in enumerate(self.probs):
            temp = torch.Tensor([t / 2**j for j, t in enumerate(bin_value[:i] + [1])])
            logit += torch.dot(temp, prob_arr) / 2**i
            prob = sigmoid(logit)
            if bin_value[i] == 0:
                prob = 1 - prob
            cumulative_probability *= prob
        return cumulative_probability.item()
    
    def get_probdist(self):
        max_val = (2 ** len(self.probs)) - 1
        return [dist.get_prob(i) for i in range(max_val)]

probs = [
    [0.1],
    [0.1, 0.01],
    [0, 0.1, 0.2],
    [0, 0, 0, 0.3],
    [0, 0, 0, 0, 0.4],
    [0, 0, 0, 0, 0, 0.7],
    [0, 0, 0, 0, 0, 0, 0.8],
    [0, 0, 0, 0, 0, 0, 0, 0.1],
]

probs = []
for i in range(8):
    probs.append([random.random() - 0.5 for r in range(i+1)])
print(probs)
dist = GeneralProbdist(probs)

for _ in range(10):
    print(dist.sample())

import plotly.graph_objects as go
import numpy as np


probdist = dist.get_probdist()

x = np.arange(len(probdist))

fig = go.Figure(data=go.Scatter(x=x, y=probdist))
fig.show()