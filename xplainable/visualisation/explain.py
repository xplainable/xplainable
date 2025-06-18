""" Copyright Xplainable Pty Ltd, 2023"""

import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt

def _generate_explain_plot_data(model, label_rounding=5):

    def get_plot_data(f):
        _profile = model.profile
        if f in _profile['numeric']:
            prof = pd.DataFrame(_profile['numeric'][f])

            if prof.empty:
                return

            prof['value'] = prof['lower'].round(label_rounding).astype(
                str) + " - " + prof['upper'].round(label_rounding).astype(str)

            prof = prof[['value', 'score', 'mean', 'freq']]

        elif f in _profile['categorical']:
            prof = pd.DataFrame(_profile['categorical'][f])
            if prof.empty:
                return

            prof = prof[['category', 'score', 'mean', 'freq']]
            prof = prof.rename(columns={'category': 'value'})
            prof = prof.explode(['value', 'mean', 'freq'])
        else:
            return

        prof['feature'] = f
        prof['score_label'] = prof['score'].apply(lambda x: str(round(x*100, 1)))
        
        prof = prof.rename(columns={'score': 'contribution', 'freq': 'frequency'})

        return prof.reset_index()

    plot_data = [get_plot_data(i) for i in model.columns]
    prof = pd.concat(
        [i for i in plot_data if i is not None]).reset_index(drop=True)

    return prof
        
def _plot_explainer(model, label_rounding=5):

    try:
        import altair as alt
    except ImportError:
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[plotting]' to use "
                          "this feature.") from None

    fi = pd.DataFrame(
            {i: {'importance': v} for i, v in model.feature_importances.items()}
            ).T.reset_index()

    fi = fi.rename(columns={'index': 'feature'})
    fi['importance_label'] = fi['importance'].apply(
        lambda x: str(round(x*100, 1))+'%')

    data = _generate_explain_plot_data(model, label_rounding)

    single = alt.selection_point(
        fields=['feature'],
        value=list(model.feature_importances.keys())[-1])
    
    brush = alt.selection_interval(encodings=['y'])
    brush2 = alt.selection_interval(encodings=['y'])
    
    # Define the dropdown selection
    dropdown = alt.selection_point(
        name='Select',
        fields=['column'],
        bind=alt.binding_select(options=['contribution', 'mean', 'frequency']),
        value='contribution'
    )

    feature_importances = alt.Chart(
        fi,
        title='Feature Importances').mark_bar(color='#0080ea').encode(
            x='importance', y=alt.Y('feature',
                sort=alt.SortField(field='importance', order='descending')),
            tooltip='importance_label',
            color=alt.condition(
            single, alt.value('lightgray'), alt.value('#0080ea'))
    ).properties(width=330, height=400).transform_filter(
        brush).add_params(single, dropdown)
    
    view = alt.Chart(fi).mark_bar(color='#0080ea').encode(
        y=alt.Y('feature:N', sort='-x', axis=alt.Axis(
        labels=False, title=None)),
        x=alt.X('importance:Q', axis=alt.Axis(labels=False, title=None))
    ).properties(height=400, width=25).add_params(brush)
    
    view2 = alt.Chart(data).mark_bar(color='#e14067').encode(
        y=alt.Y('value:N', sort=alt.SortField(field='index', order='descending'),
                axis=alt.Axis(labels=False, title=None)),
        x=alt.X('contribution:Q', axis=alt.Axis(labels=False, title=None)),
        color=alt.condition(
            alt.datum.contribution < 0,
            alt.value("#e14067"),
            alt.value("#12b980")
        )
    ).properties(height=400, width=25).add_params(brush2).transform_filter(
        single
    )

    profile = alt.Chart(data, title='Contributions').mark_bar(
        color='#e14067').encode(
        x='val:Q',
        y=alt.Y('value', sort=alt.SortField(field='index', order='descending')),
        tooltip='score_label',
        color=alt.condition(
            alt.datum.contribution < 0,
            alt.value("#e14067"),
            alt.value("#12b980")
        )
    ).transform_fold(
    ['contribution', 'mean', 'frequency'], as_=['column', 'val']
    ).transform_filter(
        dropdown
    ).properties(
            width=330,
            height=400
        ).transform_filter(
            single
        ).transform_filter(
            brush2
        ).add_params(dropdown)
    
    display(HTML("""
    <style>
    .vega-bind {
      text-align:right;
    }
    </style>
    """))

    return (feature_importances | view | profile | view2)

def _plot_feature_importances(model, feature=''):

    try:
        import altair as alt
    except ImportError:
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[plotting]' to use "
                          "this feature.") from None
    
    fi = pd.DataFrame(
            {i: {'importance': v} for i, v in model.feature_importances.items()}
            ).T.reset_index()

    fi = fi.rename(columns={'index': 'feature'})
    fi['importance_label'] = fi['importance'].apply(
        lambda x: str(round(x*100, 1))+'%')
    
    brush = alt.selection_interval(encodings=['y'])

    feature_importances = alt.Chart(
        fi,
        title='Feature Importances').mark_bar(color='#0080ea').encode(
            x='importance:Q', y=alt.Y('feature:N',
                sort=alt.SortField(field='importance:Q', order='descending')),
            tooltip='importance_label',
            color=alt.condition(
        alt.datum.feature == feature,
        alt.value('lightgray'), alt.value('#0080ea'))
        ).properties(width=175, height=300).transform_filter(brush)
    
    view = alt.Chart(fi).mark_bar(color='#0080ea').encode(
        y=alt.Y('feature:N', sort=alt.SortField(
        field='importance:Q', order='descending'), axis=alt.Axis(
        labels=False, title=None)),
        x=alt.X('importance:Q', axis=alt.Axis(labels=False, title=None))
    ).properties(height=300, width=25).add_params(brush)
    
    combined_chart = (feature_importances | view).configure(
        background='#1F2937',
        title=alt.TitleConfig(color='white'),
        axis=alt.AxisConfig(
            titleColor='white',
            labelColor='white',
            domainColor='white',
            tickColor='white'
        ),
        legend=alt.LegendConfig(
            titleColor='white',
            labelColor='white',
            strokeColor='white',
            fillColor='#1F2937',
        )
    )
    
    return combined_chart

def _plot_contributions(model, feature='__all__'):
    try:
        import altair as alt
    except ImportError:
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[plotting]' to use "
                          "this feature.") from None
    
    if feature == '__all__':
        features = list(model.columns)
    else:
        features = [feature]
    
    plot_data = _generate_explain_plot_data(model)
    filt = plot_data[plot_data['feature'].isin(features)]

    brush = alt.selection_interval(encodings=['y'])

    profile = alt.Chart(filt, title='Contributions').mark_bar(
            color='#e14067').encode(
            x='contribution:Q',
            y=alt.Y('value:N', sort=alt.SortField(field='index:Q', order='ascending')),
            tooltip='value:N', # or 'column' or 'value' based on what you want to show in tooltip
            color=alt.condition(
                alt.datum.contribution < 0,
                alt.value("#e14067"),
                alt.value("#12b980")
            )
        ).transform_fold(
            ['contribution', 'mean', 'frequency'], as_=['column', 'val']
        ).properties(
                width=175,
                height=300
        ).transform_filter(
            brush
        )
    

    view = alt.Chart(filt).mark_bar(color='#e14067').encode(
            y=alt.Y('value:N', sort=alt.SortField(field='index:Q', order='ascending'),
                    axis=alt.Axis(labels=False, title=None)),
            x=alt.X('contribution:Q', axis=alt.Axis(labels=False, title=None)),
            color=alt.condition(
                alt.datum.contribution < 0,
                alt.value("#e14067"),
                alt.value("#12b980")
            )).properties(
            width=25,
            height=300
        ).add_params(
            brush
        )
    
    combined_chart = (profile | view).configure(
        background='#1F2937',
        title=alt.TitleConfig(color='white'),
        axis=alt.AxisConfig(
            titleColor='white',
            labelColor='white',
            domainColor='white',
            tickColor='white'
        ),
        legend=alt.LegendConfig(
            titleColor='white',
            labelColor='white',
            strokeColor='white',
            fillColor='#1F2937',
        )
    )
    
    return combined_chart

def _plot_local_explainer(model, df, subsample=100):

    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[plotting]' to use "
                          "this feature.") from None
    
    buttons = []
    data = []
    
    if type(df) == pd.Series:
        df = pd.DataFrame(df).T
        
    if len(df) > 100:
        sample_size = min(len(df), subsample)
        df = df.sample(sample_size)

    for i in range(len(df)):
        row = df.iloc[i]
        base = row['base_value']
        measures = row.drop('base_value').values.tolist()
        cumulative = [float(base)] + measures + [sum([base]+measures)]
        
        trace = go.Waterfall(
            name = "explainer", orientation = "h",
            measure = ["relative"] * (len(measures)+1) + ["total"],
            y = ['Base Value'] + model.columns + ['Total'],
            textposition = "outside",
            text = [str(round(v, 4)) for v in cumulative],
            x = cumulative,
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            hovertemplate="Feature: %{y}<br>Contribution: %{text}<br>Cumulative: %{x:.4f}<extra></extra>",
            visible = False
        )

        data.append(trace)

        buttons.append(dict(label=str(df.index[i]),
                            method='update',
                            args=[{'visible': [j==i for j in range(len(df))]}]))

    # Make the first trace visible
    data[0]['visible'] = True
    fig = go.Figure(data=data, )

    fig.update_layout(
        showlegend=False,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'active': 0,
            'x': 0.33,
            'y': 1.215
        }],
        width=660,
        title='Local Explainer for: '
    )
    
    fig.show()

def create_waterfall_plot(data, title="Xplainable Waterfall Plot", figsize=(10, 8)):
    """
    Create a waterfall plot similar to SHAP waterfall plots using seaborn.
    
    Parameters:
    -----------
    data : list or dict
        Data in the format you provided with breakdown containing feature contributions
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """

    # Extract breakdown data
    if isinstance(data, list):
        breakdown = data[0]['breakdown']
    else:
        breakdown = data['breakdown']
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(breakdown)
    
    # Separate base value from other features
    base_idx = df[df['feature'] == 'base_value'].index
    if len(base_idx) > 0:
        base_value = df.loc[base_idx[0], 'score']
        df = df.drop(base_idx)
    else:
        base_value = 0
    
    # Filter out zero contributions and sort by absolute contribution
    df = df[df['score'] != 0].copy()
    df['abs_score'] = df['score'].abs()
    df = df.sort_values('abs_score', ascending=True)
    
    # Calculate cumulative values for waterfall effect
    df['cumulative'] = df['score'].cumsum() + base_value
    df['cumulative_prev'] = df['cumulative'] - df['score']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    positive_color = '#12b980'  # Green for positive contributions
    negative_color = '#e14067'  # Red for negative contributions
    base_color = '#6b7280'      # Gray for base value
    total_color = '#374151'     # Dark gray for total
    
    # Plot base value
    ax.barh(0, base_value, color=base_color, alpha=0.8, height=0.6)
    ax.text(base_value/2, 0, f'Base Value\n{base_value:.3f}', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Plot feature contributions
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        color = positive_color if row['score'] > 0 else negative_color
        
        # For positive contributions, start from previous cumulative
        # For negative contributions, start from current cumulative
        if row['score'] > 0:
            left = row['cumulative_prev']
        else:
            left = row['cumulative']
            
        ax.barh(i, abs(row['score']), left=left, color=color, alpha=0.8, height=0.6)
        
        # Add feature name and value as text
        feature_text = f"{row['feature']}\n{row['score']:+.3f}"
        if pd.notna(row['value']) and row['value'] != 'nan':
            feature_text += f"\n(value: {row['value']})"
            
        ax.text(row['cumulative'] - row['score']/2, i, feature_text, 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Plot total/final prediction
    final_value = df['cumulative'].iloc[-1] if len(df) > 0 else base_value
    ax.barh(len(df) + 1, final_value, color=total_color, alpha=0.8, height=0.6)
    ax.text(final_value/2, len(df) + 1, f'Total\n{final_value:.3f}', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Customize the plot
    ax.set_yticks(range(len(df) + 2))
    ax.set_yticklabels(['Base Value'] + df['feature'].tolist() + ['Total'])
    ax.set_xlabel('Contribution Score')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add connecting lines between bars (waterfall effect)
    for i in range(len(df)):
        if i == 0:
            # Connect base to first feature
            start_x = base_value
            end_x = df.iloc[i]['cumulative_prev']
        else:
            # Connect previous feature to current
            start_x = df.iloc[i-1]['cumulative']
            end_x = df.iloc[i]['cumulative_prev']
        
        if abs(end_x - start_x) > 1e-10:  # Only draw if there's a meaningful difference
            ax.plot([start_x, end_x], [i, i+1], 'k--', alpha=0.5, linewidth=1)
    
    # Connect last feature to total
    if len(df) > 0:
        ax.plot([df.iloc[-1]['cumulative'], final_value], 
                [len(df), len(df) + 1], 'k--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return fig
