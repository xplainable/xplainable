""" Copyright Xplainable Pty Ltd, 2023"""

import pandas as pd
from IPython.display import clear_output
import ipywidgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..components import CardWidget
from ...utils.handlers import add_thousands_separator
        

class ScenarioClassification:
    
    def __init__(self, model, ranges={}):
        
        self.model = model
        self.ranges = ranges
        
        self.widgets = {}
        self.feature_filter = widgets.IntRangeSlider(
                            min=1,
                            step=1,
                            orientation='vertical'
                        )
        self.toggle_sort = widgets.ToggleButton(value=True, description='sort')
        self.waterfall_output = widgets.Output()
        self.calibration_output = widgets.Output()
        
        self.score = None
        self.proba_card = CardWidget("PROBABILITY", "-")
        self.score_card = CardWidget("SCORE", "-")
        self.multi_card = CardWidget("MULTIPLIER", "-")
        self.support_card = CardWidget("SUPPORT", "-")
        
    def build_numeric_widget(self, feature, rng={}):
        
        if len(rng) > 0:
            minn = rng['lower']
            maxx = rng['upper']
        else:
            minn = feature[0]['upper'] - feature[0]['upper'] * 0.01
            maxx = feature[-1]['lower'] + feature[-1]['lower'] * 0.01
        
        v = minn + (maxx - minn) / 2
        
        if len(rng) > 0 and rng['type'] == int:
            w = widgets.IntSlider(min=minn, max=maxx, value=v)
        else:
            w = widgets.FloatSlider(min=minn, max=maxx, value=v)
        
        w.layout = widgets.Layout(max_width='250px', display='flex')
        
        def on_change(_):

            with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()
            
        w.observe(on_change, names=['value'])
        
        return w
    
    def build_categorical_widget(self, feature):
        
        cats = []
        for node in feature:
            cats += node['category']
            
        w = widgets.Dropdown(options=cats, value=cats[0])
        w.layout = widgets.Layout(max_width='150px', display='flex')
        
        def on_change(_):
            
            with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()

        w.observe(on_change, names=['value'])
        
        return w
    
    def generate_visuals(self):
        
        p, s, m = self.get_predictions()
        support = self.model._support_map[int(s)]

        self.proba_card.set_value(f'{round(p, 2)}%')
        self.score_card.set_value(s)
        self.multi_card.set_value(f'{m}x')
        self.support_card.set_value(support)

        self.score, metadata = self.get_prediction_breakdown()
        
        total = sum([v for i, v in metadata.items()])
        metadata['total'] = total
        length = len(metadata)
        
        df = pd.DataFrame.from_dict(
            metadata, orient='index', columns=['contribution'])

        df['abs_contribution'] = abs(df['contribution'])
        if self.toggle_sort.value:
            slc = df.iloc[1:-1].copy()
            slc = slc.sort_values('abs_contribution', ascending=False)
            df = pd.concat(
                [pd.DataFrame(df.iloc[0].copy()).T,
                slc,
                pd.DataFrame(df.iloc[length-1].copy()).T]
                )
        
        df.index.name = 'feature'
        df.reset_index(inplace=True)
        
        # Apply filtering
        ll, uu = self.feature_filter.value
        u = abs(ll - self.feature_filter.max - 1)
        l = abs(uu - self.feature_filter.max - 1)
        _filter = [0]+list(np.arange(l, u+1))
        _remainder = [i for i in range(length-1) if i not in _filter]
        
        if len(_remainder) > 0:
            remainder_df = pd.DataFrame(df.iloc[_remainder].sum(axis=0)).T
            remainder_df['feature'] = 'other'
            data = pd.concat(
                [df.iloc[_filter].copy(),
                remainder_df,
                pd.DataFrame(df.iloc[length-1].copy()).T],
                ignore_index=True).reset_index(drop=True)
        
        else:
            data = df.copy()

        length = len(data)
        
        data['total'] = data['contribution'].cumsum()
        data['total2']= data['total'].shift(1).fillna(0)
        data['lower'] = data[['total','total2']].min(axis=1)
        data['upper'] = data[['total','total2']].max(axis=1)

        data['colour'] = data['contribution'].apply(
            lambda x: '#12b980' if x > 0 else '#e14067')

        data.loc[length-1, 'lower'] = 0
        data.loc[length-1, 'upper'] = data.loc[length-1, 'contribution']
        data.loc[length-1, 'colour'] = '#0080ea'
        data.loc[0, 'colour'] = '#98999b'

        # Create Plot
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(
            2, 1, sharex = True,
            figsize=(6, 12),
            gridspec_kw={
                        'height_ratios': [6, 3]
            })
        
        bars = sns.barplot(
            data=data, y='feature', x='upper', orient='horizintal',
            palette=data['colour'], ax=ax[0])

        bars2 = sns.barplot(
            data=data, y='feature', x='lower', orient='horizintal',
            color='white', ax=ax[0])

        for i, v in enumerate(data['upper']):
            ax[0].text(
                v + 0.005, i + 0.1, round(data['contribution'][i]*100, 2))

        ax[0].set_xlim(0, 1)
        sns.despine()
        
        ax[0].set_xlabel('Contribution', size=14)
        ax[0].set_ylabel('Feature', size=14)
        
        # CALIBRATION
        plot_data = pd.DataFrame(
            self.model._calibration_map.items(),
            columns=['score', 'probability'])

        plot_data['score'] = plot_data['score'] / 100
        support_data = pd.DataFrame(
            self.model._support_map.items(), columns=['score', 'support'])

        support_data['score'] = support_data['score'] / 100
        
        h = self.model._calibration_map[int(self.score*100)]
        s = self.model._support_map[int(self.score*100)]
        
        cp_line = sns.lineplot(
            data=plot_data, y='probability', x='score', ax=ax[1])
        
        plt.vlines(x=self.score, ymin=0, ymax=1, ls='--', color='#374151', lw=1)
        plt.hlines(y=h, xmin=0, xmax=1, ls='--', color='#374151', lw=1)
        plt.plot([0, 1], [0, 1], ls=':', color='#0080ea', lw=1)
        plt.text(1.05, h*0.96, f'prob: {round(h*100, 2)}%\nsupport: {s}')

        cp_line.set_xlabel('Score', size=14)
        cp_line.set_ylabel('Probability', size=14)
        
        
        cp2 = ax[1].twinx()
        axx = sns.lineplot(data=support_data, x='score', y='support',
        ax=cp2, color='#0080ea', alpha=0.1)
        
        l1 = cp2.lines[0]
        x1 = l1.get_xydata()[:,0]
        y1 = l1.get_xydata()[:,1]
        cp2.fill_between(x1,y1, color="#0080ea", alpha=0.1)
        cp2.set(yticklabels=[])
        cp2.set(ylabel=None)
        cp2.tick_params(right=False)

        return fig

    def get_feature_values(self):
        vals = {c: self.widgets[c].value if c in self.widgets \
            else 0 for c in self.model.columns}
        return vals

    def get_predictions(self):
        
        data = pd.DataFrame(
            self.get_feature_values().items(),
            columns=['Feature', 'Value']).set_index('Feature').T
                
        proba = self.model.predict_proba(data)[0]
        multi = round(proba / self.model.base_value, 2)
        proba = round(proba*100, 2)
        
        score = round(self.model.predict_score(data)[0]*100, 2)
        
        return proba, score, multi

    def get_prediction_breakdown(self):
       
        profile = self.model.profile
        
        score = self.model.base_value
        breakdown = {'base_value': self.model.base_value}
        
        for f, nodes in profile['numeric'].items():
            if f not in self.widgets:
                continue
                
            val = self.widgets[f].value
            
            leaf = [l for l in nodes if val <= l['upper']][0]
            
            score += leaf['score']
            breakdown[f] = leaf['score']
            
        for f, nodes in profile['categorical'].items():
            if f not in self.widgets:
                continue
                
            val = self.widgets[f].value
            
            leaf = [l for l in nodes if val == l['category']][0]
            
            score += leaf['score']
            breakdown[f] = leaf['score']
                
        return score, breakdown
    
    def run(self):
        
        profile = self.model.profile
        
        for f in profile['categorical']:
            feature = profile['categorical'][f]
            if len(feature) == 0:
                continue
            w = self.build_categorical_widget(feature)
            self.widgets[f] = w
        
        for f in profile['numeric']:
            feature = profile['numeric'][f]
            if len(feature) == 0:
                continue
            rng = self.ranges[f] if f in self.ranges else {}
            w = self.build_numeric_widget(feature, rng)
            self.widgets[f] = w
        
        scenario_title = widgets.HTML('<h3>Create Scenario</h3>')
        divider = widgets.HTML(f'<hr class="solid">')
        
        selector_children = []
        
        for f in list(
            profile['categorical'].keys())+list(profile['numeric'].keys()):

            if f not in self.widgets:
                continue
            title = widgets.HTML(f'<h5>{f}</h5>')
            w = self.widgets[f]
            box = widgets.VBox([title, w])
            box.layout = widgets.Layout(
                overflow='hidden', flex='none', max_width='250px')
            selector_children.append(box)
        
        selectors = widgets.VBox(selector_children)
        selectors.layout = widgets.Layout(
            height='700px', flex='none', overflow_y='scroll')
        
        scenario_column = widgets.VBox([scenario_title, divider, selectors])
        
        plot_title = widgets.HTML('<h4>Contributions</h4>')

        cards = widgets.HBox([
            self.proba_card.card,
            self.score_card.card,
            self.multi_card.card,
            self.support_card.card
            ])
            
        waterfall_widgets = widgets.VBox([
            self.toggle_sort,
            self.feature_filter
        ])

        waterfall_widgets.layout = widgets.Layout(
            flex='none', overflow='hidden')

        waterfall_zone = widgets.HBox([
            widgets.VBox([plot_title, self.waterfall_output]),
            waterfall_widgets
        ])

        def on_filter_change(_):
            with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()
        
        self.feature_filter.max = len(self.widgets)
        self.feature_filter.value = (max(
            [0, len(self.widgets)-10]), len(self.widgets))

        self.feature_filter.observe(on_filter_change, names=['value'])
        
        self.toggle_sort.observe(on_filter_change, names=['value'])
        self.toggle_sort.layout = widgets.Layout(
            height='25px', width='45px', margin='0 0 20px 10px')
        
        output = widgets.VBox(
            [cards, divider, waterfall_zone])

        vertical_divider = widgets.HTML(
            value='''<div style="border-left: 1px solid #cccccc;
                height: 100%; margin: 0 8px;"></div>''',
            layout=widgets.Layout(height='auto', margin='0 0 0 10px')
        )

        screen = widgets.HBox([scenario_column, vertical_divider, output])
        
        with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()
        
        return screen


class ScenarioRegression:
    
    def __init__(self, model, ranges={}):
        
        self.model = model
        self.model_type = model.__class__.__name__
        self.ranges = ranges
        
        self.widgets = {}
        self.feature_filter = widgets.IntRangeSlider(
                            min=1,
                            step=1,
                            orientation='vertical'
                        )
        self.toggle_sort = widgets.ToggleButton(value=True, description='sort')
        self.waterfall_output = widgets.Output()
        
        self.score = None
        self.prediction_card = CardWidget("PREDICTION", "-")
        
    def build_numeric_widget(self, feature, rng={}):
        
        if len(rng) > 0:
            minn = rng['lower']
            maxx = rng['upper']
        else:
            minn = feature[0]['upper'] - feature[0]['upper'] * 0.01
            maxx = feature[-1]['lower'] + feature[-1]['lower'] * 0.01
        
        v = minn + (maxx - minn) / 2
        
        if len(rng) > 0 and rng['type'] == int:
            w = widgets.IntSlider(min=minn, max=maxx, value=v)
        else:
            w = widgets.FloatSlider(min=minn, max=maxx, value=v)
        
        w.layout = widgets.Layout(max_width='250px', display='flex')
        
        def on_change(_):

            with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()
            
        w.observe(on_change, names=['value'])
        
        return w
    
    def build_categorical_widget(self, feature):
        
        cats = []
        for node in feature:
            cats += node['category']
            
        w = widgets.Dropdown(options=cats, value=cats[0])
        w.layout = widgets.Layout(max_width='150px', display='flex')
        
        def on_change(_):
            
            with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()

        w.observe(on_change, names=['value'])
        
        return w
    
    def generate_visuals(self):
        

        pred = self.get_predictions()
        self.prediction_card.set_value(pred)

        self.score, metadata = self.get_prediction_breakdown()
        
        total = sum([v for i, v in metadata.items()])
        metadata['total'] = total
        length = len(metadata)
        
        df = pd.DataFrame.from_dict(
            metadata, orient='index', columns=['contribution'])

        df['abs_contribution'] = abs(df['contribution'])
        if self.toggle_sort.value:
            slc = df.iloc[1:-1].copy()
            slc = slc.sort_values('abs_contribution', ascending=False)
            df = pd.concat(
                [pd.DataFrame(df.iloc[0].copy()).T,
                slc,
                pd.DataFrame(df.iloc[length-1].copy()).T]
                )
        
        df.index.name = 'feature'
        df.reset_index(inplace=True)
        
        # Apply filtering
        ll, uu = self.feature_filter.value
        u = abs(ll - self.feature_filter.max - 1)
        l = abs(uu - self.feature_filter.max - 1)
        _filter = [0]+list(np.arange(l, u+1))
        _remainder = [i for i in range(length-1) if i not in _filter]
        
        if len(_remainder) > 0:
            remainder_df = pd.DataFrame(df.iloc[_remainder].sum(axis=0)).T
            remainder_df['feature'] = 'other'
            data = pd.concat(
                [df.iloc[_filter].copy(),
                remainder_df,
                pd.DataFrame(df.iloc[length-1].copy()).T],
                ignore_index=True).reset_index(drop=True)
        
        else:
            data = df.copy()

        length = len(data)
        
        data['total'] = data['contribution'].cumsum()
        data['total2']= data['total'].shift(1).fillna(0)
        data['lower'] = data[['total','total2']].min(axis=1)
        data['upper'] = data[['total','total2']].max(axis=1)

        data['colour'] = data['contribution'].apply(
            lambda x: '#12b980' if x > 0 else '#e14067')

        data.loc[length-1, 'lower'] = 0
        data.loc[length-1, 'upper'] = data.loc[length-1, 'contribution']
        data.loc[length-1, 'colour'] = '#0080ea'
        data.loc[0, 'colour'] = '#98999b'

        # Create Plot
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(
            figsize=(6, 12)
            )
        
        bars = sns.barplot(
            data=data, y='feature', x='upper', orient='horizontal',
            palette=data['colour'])

        bars2 = sns.barplot(
            data=data, y='feature', x='lower', orient='horizontal',
            color='white')
        
        try:
            for i, v in enumerate(data['upper']):
                ax.text(v + 0.005, i + 0.1, round(
                    data['contribution'][i], 2))
        except Exception:
            pass

        sns.despine()
        
        ax.set_xlabel('Contribution', size=14)
        ax.set_ylabel('Feature', size=14)
        
        return fig

    def get_feature_values(self):
        vals = {c: self.widgets[c].value if c in self.widgets \
            else 0 for c in self.model.columns}

        return vals

    def get_predictions(self):
        
        data = pd.DataFrame(
            self.get_feature_values().items(),
            columns=['Feature', 'Value']).set_index('Feature').T
                
        score = round(self.model.predict(data)[0], 2)
        
        return score

    def get_prediction_breakdown(self):
       
        profile = self.model.profile
        
        score = self.model.base_value
        breakdown = {'base_value': self.model.base_value}
        
        for f, nodes in profile['numeric'].items():
            if f not in self.widgets:
                continue
                
            val = self.widgets[f].value
            
            leaf = [l for l in nodes if val <= l['upper']][0]
            
            score += leaf['score']
            breakdown[f] = leaf['score']
            
        for f, nodes in profile['categorical'].items():
            if f not in self.widgets:
                continue
                
            val = self.widgets[f].value
            
            leaf = [l for l in nodes if val == l['category']][0]
            
            score += leaf['score']
            breakdown[f] = leaf['score']
                
        return score, breakdown
    
    def run(self):
        
        profile = self.model.profile
        
        for f in profile['categorical']:
            feature = profile['categorical'][f]
            if len(feature) == 0:
                continue
            w = self.build_categorical_widget(feature)
            self.widgets[f] = w
        
        for f in profile['numeric']:
            feature = profile['numeric'][f]
            if len(feature) == 0:
                continue
            rng = self.ranges[f] if f in self.ranges else {}
            w = self.build_numeric_widget(feature, rng)
            self.widgets[f] = w
        
        scenario_title = widgets.HTML('<h3>Create Scenario</h3>')
        divider = widgets.HTML(f'<hr class="solid">')
        
        selector_children = []
        
        for f in list(
            profile['categorical'].keys())+list(profile['numeric'].keys()):

            if f not in self.widgets:
                continue
            title = widgets.HTML(f'<h5>{f}</h5>')
            w = self.widgets[f]
            box = widgets.VBox([title, w])
            box.layout = widgets.Layout(
                overflow='hidden', flex='none', max_width='250px')
            selector_children.append(box)
        
        selectors = widgets.VBox(selector_children)
        selectors.layout = widgets.Layout(
            height='750px', flex='none', overflow_y='scroll')
        
        scenario_column = widgets.VBox([scenario_title, divider, selectors])
        
        plot_title = widgets.HTML('<h4>Contributions</h4>')
        
        cards = widgets.HBox([
            self.prediction_card.card
            ])
            
        waterfall_widgets = widgets.VBox([
            self.toggle_sort,
            self.feature_filter
        ])

        waterfall_widgets.layout = widgets.Layout(
            flex='none', overflow='hidden')

        waterfall_zone = widgets.HBox([
            widgets.VBox([plot_title, self.waterfall_output]),
            waterfall_widgets
        ])

        def on_filter_change(_):
            with self.waterfall_output:
                clear_output(wait=True)
                fig1 = self.generate_visuals()
                plt.show()
        
        self.feature_filter.max = len(self.widgets)
        self.feature_filter.value = (max(
            [0, len(self.widgets)-10]), len(self.widgets))
        
        self.feature_filter.observe(on_filter_change, names=['value'])
        
        self.toggle_sort.observe(on_filter_change, names=['value'])
        self.toggle_sort.layout = widgets.Layout(
            height='25px', width='45px', margin='0 0 20px 10px')
        
        output = widgets.VBox(
            [cards, divider, waterfall_zone])

        vertical_divider = widgets.HTML(
            value='''<div style="border-left: 1px solid #cccccc;
                height: 100%; margin: 0 8px;"></div>''',
            layout=widgets.Layout(height='auto', margin='0 0 0 10px')
        )

        screen = widgets.HBox([scenario_column, vertical_divider, output])
        
        with self.waterfall_output:
            clear_output(wait=True)
            fig1 = self.generate_visuals()
            plt.show()
        
        return screen