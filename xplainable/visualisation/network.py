import networkx as nx
import numpy as np

def plot_network_graphs(graphs, animate=True, index=0):
    """ Plot a list of networkx graphs using plotly.
    
    Args:
        graphs (list): A list of networkx graphs.
        animate (bool): Whether to animate the graph.
        index (int): The index of the graph to plot.

    Returns:
        A plotly figure.
    """

    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[plotting]' to use "
                          "this feature.") from None

    if len(graphs) == 0:
        return None

    frames = []
    for i, G in enumerate(graphs):
        # 3d spring layout
        pos = nx.spring_layout(G, dim=3, seed=779)
        # Extract node and edge positions from the layout
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        edge_trace = go.Scatter3d(x=[], y=[], z=[], mode='lines')
        for edge in edge_xyz:
            x0, y0, z0 = edge[0]
            x1, y1, z1 = edge[1]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            edge_trace['z'] += tuple([z0, z1, None])

        node_trace = go.Scatter3d(
            x=node_xyz[:,0], y=node_xyz[:,1], z=node_xyz[:,2],
            mode='markers+text',
            marker=dict(size=6, color='rgb(22, 96, 167)'),
            text=list(G.nodes),
            textposition="top center")

        frames.append(go.Frame(data=[edge_trace, node_trace], name=f'frame{i}'))

    layout = go.Layout(showlegend=False, 
                       scene=dict(aspectmode="cube",
                                  xaxis=dict(showticklabels=False),
                                  yaxis=dict(showticklabels=False),
                                  zaxis=dict(showticklabels=False)),
                       margin=dict(t=0, l=0, r=0, b=0))

    if animate:
        layout.updatemenus=[
            dict(type='buttons',
                 showactive=False,
                 buttons=[dict(label='Play',
                               method='animate',
                               args=[None, 
                                     dict(frame=dict(duration=500, 
                                                     redraw=True), 
                                                     fromcurrent=True, 
                                                     transition=dict(duration=0)
                                                     )])])]
        
        fig = go.Figure(data=frames[0]['data'], layout=layout, frames=frames)
    else:
        fig = go.Figure(data=frames[index]['data'], layout=layout)

    fig.show()
