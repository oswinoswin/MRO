import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


my_data = pd.read_csv('data/eggs6.csv')
x = my_data['points']
y = my_data['error']
print('{}'.format(y))
plt.plot(x, y)
plt.show()