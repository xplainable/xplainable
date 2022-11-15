import altair as alt
import pandas as pd

def _generate_explain_plot_data(model):

        def get_plot_data(f):
            if f in model._profile['numeric']:
                prof = pd.DataFrame({i: v for i, v in model._profile['numeric'][f].items() if len(v) > 0}).T.reset_index()
                if prof.empty:
                    return
                prof['value'] = prof['lower'].astype(str) + " - " + prof['upper'].astype(str)
                prof = prof[['value', 'score']]
            elif f in model._profile['categorical']:
                prof = pd.DataFrame({i: v for i, v in model._profile['categorical'][f].items() if len(v) > 0}).T.reset_index()
                if prof.empty:
                    return
                prof = prof[['categories', 'score']]
                prof['categories'] = prof['categories'].apply(lambda x: list(x))
                prof = prof.rename(columns={'categories': 'value'})
                prof = prof.explode('value')
            else:
                return

            prof['feature'] = f
            prof['score_label'] = prof['score'].apply(lambda x: str(round(x*100, 1))+'%')

            return prof.reset_index()
        
        fimp = pd.DataFrame({i: {'importance': v} for i, v in model._feature_importances.items()}).T.reset_index()
        fimp = fimp.rename(columns={'index': 'feature'})
        fimp['importance_label'] = fimp['importance'].apply(lambda x: str(round(x*100, 1))+'%')
        plot_data = [get_plot_data(i) for i in model._feature_importances.keys()]
        prof = pd.concat([i for i in plot_data if i is not None]).reset_index(drop=True)
        
        return fimp, prof

def generate_explain_plots(model):
    fi, p = _generate_explain_plot_data(model)
    
    single = alt.selection(type="single", fields=['feature'], init={'feature': list(model._feature_importances.keys())[0]})

    feature_importances = alt.Chart(fi, title='Feature Importances').mark_bar(color='#0080ea').encode(
        x='importance',
        y='feature',
        tooltip='importance_label',
        color=alt.condition(single, alt.value('lightgray'), alt.value('#0080ea'))
    ).properties(
        width=350,
        height=400
    ).add_selection(
        single
    )

    profile = alt.Chart(p, title='Contributions').mark_bar(color='#e14067').encode(
        x='score',
        y=alt.Y('value', sort=alt.SortField(field='index', order='descending')),
        tooltip='score_label',
        color=alt.condition(
            alt.datum.score < 0,
            alt.value("#e14067"),
            alt.value("#12b980")
        )
    ).properties(
        width=400,
        height=400
    ).transform_filter(
        single
    )


    return (feature_importances | profile)