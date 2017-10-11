import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


my_data = pd.read_csv('data/eggs.csv')
x = my_data['points']
y = my_data['value']
print('{}'.format(y))
plt.plot(x, y)
plt.show()