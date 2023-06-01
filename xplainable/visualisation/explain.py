""" Copyright Xplainable Pty Ltd, 2023"""

import altair as alt
import pandas as pd
from IPython.display import HTML

def _generate_explain_plot_data(model):

    def get_plot_data(f):
        _profile = model.profile
        if f in _profile['numeric']:
            prof = pd.DataFrame(_profile['numeric'][f])

            if prof.empty:
                return

            prof['value'] = prof['lower'].round(3).astype(str) + " - " + \
                prof['upper'].round(3).astype(str)

            prof = prof[['value', 'score', 'mean', 'freq']]

        elif f in _profile['categorical']:
            prof = pd.DataFrame(_profile['categorical'][f])

            if prof.empty:
                return

            prof['mean'] = prof['means']
            prof['freq'] = prof['frequencies']
            prof = prof[['categories', 'score', 'mean', 'freq']]
            prof = prof.rename(columns={'categories': 'value'})
            prof = prof.explode(['value', 'mean', 'freq'])

        else:
            return

        prof['feature'] = f
        prof['score_label'] = prof['score'].apply(
            lambda x: str(round(x*100, 1)))
        
        prof = prof.rename(columns={'score': 'contribution', 'freq': 'frequency'})

        return prof.reset_index()

    plot_data = [get_plot_data(i) for i in model.columns]
    prof = pd.concat(
        [i for i in plot_data if i is not None]).reset_index(drop=True)

    return prof
        
def _plot_explainer(model):

    fi = pd.DataFrame(
            {i: {'importance': v} for i, v in model.feature_importances.items()}
            ).T.reset_index()

    fi = fi.rename(columns={'index': 'feature'})
    fi['importance_label'] = fi['importance'].apply(
        lambda x: str(round(x*100, 1))+'%')

    data = _generate_explain_plot_data(model)

    single = alt.selection_single(
        fields=['feature'],
        value=list(model.feature_importances.keys())[-1])
    
    brush = alt.selection_interval(encodings=['y'])
    brush2 = alt.selection_interval(encodings=['y'])
    
    # Define the dropdown selection
    dropdown = alt.selection_single(
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
        brush).add_selection(single, dropdown)
    
    view = alt.Chart(fi).mark_bar().encode(
        y=alt.Y('feature:N', sort='-x', axis=alt.Axis(
        labels=False, title=None)),
        x=alt.X('importance:Q', axis=alt.Axis(labels=False, title=None))
    ).properties(height=400, width=25).add_selection(brush)
    
    view2 = alt.Chart(data).mark_bar().encode(
        y=alt.Y('value:N', sort='-x', axis=alt.Axis(labels=False, title=None)),
        x=alt.X('contribution:Q', axis=alt.Axis(labels=False, title=None))
    ).properties(height=400, width=25).add_selection(brush2).transform_filter(
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
        ).add_selection(dropdown)
    
    display(HTML("""
    <style>
    .vega-bind {
      text-align:right;
    }
    </style>
    """))

    return (feature_importances | view | profile | view2)
