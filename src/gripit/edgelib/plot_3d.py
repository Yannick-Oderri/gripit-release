from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from future import standard_library
standard_library.install_aliases()
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='jzhu1', api_key='dHJj9eRDVTfTiOYZmtbf')

x= np.load("saveX.npy")
y = np.load("saveY.npy")
z = np.load("saveZ.npy")
trace0 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
     marker=dict(
        size=2,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

cx= np.load("save_cX.npy")
cy = np.load("save_cY.npy")
cz = np.load("save_cZ.npy")
trace1 = go.Scatter3d(
    x=cx,
    y=cy,
    z=cz,
    mode='markers',
     marker=dict(
        size=2,
        color=z,                # set color to an array/list of desired values
        colorscale='Blues',   # choose a colorscale
        opacity=0.8
    ),
    line=dict(
        color='#1f77b4',
        width=1
    )
)

x2= np.load("save_pairsX.npy")
y2 = np.load("save_pairsY.npy")
z2 = np.load("save_pairsZ.npy")
trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
     marker=dict(
        size=4,
        color=z,                # set color to an array/list of desired values
        colorscale='Portland',   # choose a colorscale
        opacity=0.8
    ),
)

x3 = np.load("save_mX.npy")
y3 = np.load("save_mY.npy")
z3 = np.load("save_mZ.npy")
trace3 = go.Scatter3d(
    x=x3,
    y=y3,
    z=z3,
    mode='markers',
     marker=dict(
        size=4,
        color=z,                # set color to an array/list of desired values
        colorscale='Picnic',   # choose a colorscale
        opacity=0.8
    ),
)


x4 = np.load("new_x.npy")
y4 = np.load("new_y.npy")
z4 = np.load("new_z.npy")
trace4 = go.Scatter3d(
    x=x4,
    y=y4,
    z=z4,
    mode='markers',
     marker=dict(
        size=4,
        color=z,                # set color to an array/list of desired values
        colorscale='Picnic',   # choose a colorscale
        opacity=0.8
    ),
)





data = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(
    title='Point Cloud',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)

name = 'eye = (x:0.1, y:0.1, z:2.5)'
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-0.1, y=-2.5, z=-0.1)
)

fig['layout'].update(
    scene=dict(camera=camera),
    title=name
)
py.iplot(fig, validate=False, filename=name)

py.iplot(fig, filename='simple-3d-scatter')