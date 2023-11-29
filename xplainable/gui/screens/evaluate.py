""" Copyright Xplainable Pty Ltd, 2023"""

from ...gui.components.bars import BarGroup
from ...utils.activation import flex_activation
from .scenario import ScenarioClassification, ScenarioRegression
from ...gui.components.tables import KeyValueTable

import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from ...core.optimisation.genetic import XEvolutionaryNetwork
from ...core.optimisation.layers import Tighten, Evolve
from ...callbacks.optimisation import RegressionCallback
from ...visualisation.explain import _generate_explain_plot_data as gen_x_plot_data


class EvaluateClassifier:
    
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.initialise()

        self.y_val = None
        self.y_val_prob = None

        self.action_buttons = []
        self.scenarios = None
        self.ranges = {}
        
    def initialise(self):
        
        self._feature_importance_chart = None
        
        self._calibration_display = None
        
        fimp = self.model.feature_importances
        cols = list(fimp.keys())
        cols.reverse()
        cols.append(None)
        
        self.feature_selector = widgets.Dropdown(
            options=cols)
        
        self.threshold = widgets.FloatSlider(
            style={'description_width': 'initial'},
            min=0.01, max=1.0, step=0.01, value=0.5)
        
        self.profile_plot_data = self._generate_explain_plot_data()
        self.feature_importance_bars = None
        self.calibration_box = None
        self.calibration_layout = widgets.Layout(
            display='none',
            padding='20px',
            margin='0 0 0 50px',
            width = '350px'
        )
        
        # update calibration smoothing
        self.smoother = widgets.IntSlider(
            description='smoother range',
            style={'description_width': 'initial'},
            value=15, min=5, max=30, step=1)
        self.threshold.layout.width = '410px'

        # Encode y if required
        if len(self.model.target_map) and self.y.dtype == 'O':
            self.y = self.y.map(self.model.target_map.forward)
        
        def on_smoother_change(_):
            self.model._calibration_map = self.model._map_calibration(
                self.y, self.y_prob, self.smoother.value)
            self._refresh()
            
        self.smoother.observe(on_smoother_change, names=['value'])

        def on_feature_change(_):
            
            f = self.feature_selector.value
            
            if f is None:
                return
            
            #params = self.model.feature_params[f]
            params = self.model._constructs[self.model.columns.index(f)].params
            
            self.calibration_box.children[1].value = params.max_depth
            self.calibration_box.children[2].value = params.min_info_gain
            self.calibration_box.children[3].value = params.min_leaf_size
            self.calibration_box.children[4].value = params.weight
            self.calibration_box.children[5].value = params.power_degree
            self.calibration_box.children[6].value = params.sigmoid_exponent
        
        self.feature_selector.observe(on_feature_change)

        self.y_prob = self.model.predict_score(self.X)

    def _generate_explain_plot_data(self):
        return gen_x_plot_data(self.model, 3)

    def _generate_feature_importance(self):
        def explore(b):
            self.feature_selector.value = b.id

        fimp = self.model.feature_importances
        cols = list(fimp.keys())
        cols.reverse()

        group = BarGroup(items=cols)
        group.set_suffix(items=None, suffix='%')
        group.add_button(text='view', on_click=explore)
        group.displays_layout.flex = 'none'

        width = int(max([len(i) for i in self.model.columns]) * 7)

        group.label_layout.width = f'{width}px'

        for i, v in fimp.items():
            val = round(v*100, 2)
            group.set_value(i, val)

        return group.show()
    
    def _evaluation_screen(self, X, y):
        
        group = BarGroup([
            'f1-score', 'precision', 'recall', 'accuracy',
            'roc_auc', 'log_loss', 'neg_brier_loss'])
        
        group.bar_layout.width = '275px'
        group.set_bounds(['neg_brier_loss'], 0, 1)
        group.label_layout.width = f'85px'
        
        evaluation = self.model.evaluate(X, y, threshold=0.5)
        report = evaluation['classification_report']
        
        toggle_options = list(report.keys())
        toggle_options.remove('accuracy')
        toggle_buttons = widgets.ToggleButtons(options=toggle_options)
        toggle_buttons.style.button_width = '98px'
        toggle_buttons.layout = widgets.Layout(margin='10px 0 35px 0')

        # Define a colormap that smoothly transitions between the two colors
        cmap = LinearSegmentedColormap.from_list(
            'custom', ['#eeeeee', '#0080ea'], N=100, gamma=0.4)

        def _plot(
                threshold=self.threshold,
                chart_type = widgets.ToggleButtons(options=['Confusion Matrix',
                                                            'Probability']),
                metric_type=toggle_buttons):

            evaluation = self.model.evaluate(X, y, threshold=threshold)

            cm = evaluation['confusion_matrix']

            report = evaluation['classification_report']

            [group.set_value(i, round(v*100,2)) for i, v in report[
                toggle_buttons.value].items() if i != 'support']

            group.set_value('accuracy', round(report['accuracy']*100, 2))
            group.set_value('roc_auc', round(evaluation['roc_auc']*100, 2))

            group.set_value('neg_brier_loss', round(
                evaluation['neg_brier_loss'], 4))
            
            group.set_value('log_loss', round(evaluation['log_loss'], 2))
            
            if chart_type == 'Confusion Matrix':
                
                sns.set(font_scale=1.5)
                fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
                ax1 = sns.heatmap(
                    cm, annot=True, linewidths=3, cbar=False, fmt='', cmap=cmap)

                # Set the plot title and labels
                ax1.set_xlabel('Predicted label', size=14)
                ax1.set_ylabel('True label', size=14)

                # Show the plot
                plt.show()
            
            else:
                custom_params = {"axes.spines.right": False,
                                 "axes.spines.top": False}
                
                sns.set_theme(style="ticks", rc=custom_params)

                fig, ax = plt.subplots(figsize=(5.7, 4))
                plot_data = pd.DataFrame(
                    self.model._calibration_map.items(),
                    columns=['score', 'probability'])

                ax1 = sns.lineplot(
                    data=plot_data, y='probability', x='score', ax=ax)

                v = int(threshold*100)
                plt.vlines(x=v, ymin=0, ymax=1, ls='--', color='#374151', lw=1)
                plt.hlines(y=self.model.base_value, xmin=0, xmax=100, ls=':',
                           color='#374151', lw=1)

                plt.plot([0, 100], [0, 1], ls=':', color='#0080ea', lw=1)

                ax2 = ax.twinx()
                bar_data = pd.DataFrame({
                    'proba': self.y_val_prob*100, 'true': y.values})

                palette=["#e14067", "#0080ea"]
                axx = sns.histplot(
                    data=bar_data, x='proba', hue='true', bins=20, ax=ax2,
                    palette=palette)

                ax1.set_xlabel('Score', size=14)
                ax1.set_ylabel('Probability', size=14)

                plt.show()

        cm_output = widgets.Output()
        
        chart = interactive(_plot)
        
        chart.children[0].layout.width = '410px'
        
        chart.children[1].layout = widgets.Layout(margin='0 0 0 55px')
        chart.children[1].description = ''
        chart.children[2].description = ''
        
        chart.children = (
            chart.children[1],
            chart.children[0],
            self.smoother,
            chart.children[3])
        
        with cm_output:
            display(widgets.HTML('<h4>Calibration</h4>'))
            display(chart)
        
        bar_title = widgets.HTML('<h4>Performance</h4>')
        
        bar_columns = widgets.VBox([
            bar_title,
            toggle_buttons,
            group.show()
        ])
        
        bar_columns.layout = widgets.Layout(
            margin='8px 0 0 25px')
        
        screen = widgets.HBox([
            cm_output,
            bar_columns
        ])
        
        return screen
    
    def _generate_contribution_plot(self):
        
        cf1 = lambda x: '#e14067' if x < 0 else '#12b980'
        cf2 = lambda x: '#0080ea'
        
        
        plot_selector = widgets.ToggleButtons(options=['contribution', 'mean', 'frequency'])
        plot_selector.style.button_width = '102px'
        plot_selector.layout = widgets.Layout(margin='10px 0 0 90px')
        
        def _plot(feature=self.feature_selector, plot=plot_selector):
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="ticks", rc=custom_params)
            
            if feature is None:
                return
            
            plot_data = self.profile_plot_data[self.profile_plot_data['feature'] == feature]

            if plot_data.empty:
                print('')
                return
            
            cf = cf1 if plot == 'contribution' else cf2
            cp = [cf(value) for value in plot_data[plot]]
            
            fig, ax = plt.subplots(figsize=(5.7,5.3))
            
            ax1 = sns.barplot(
                data=plot_data,
                x=plot,
                y='value',
                palette=cp
            )
            
            ax1.tick_params(axis='x', labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            ax1.set_xlabel(plot.title(), size=14)
            ax1.set_ylabel('Value / Range', size=14)
            
            if plot == 'mean':
                base = self.model.base_value
                plt.axvline(x=base, ls=':', color='#374151', lw=1)

            plt.title = 'Contributions'

            plt.show()
            
        plot = interactive(_plot)
        
        plot_selector.description = ''
        
        recalibrate = widgets.ToggleButton(description='Recalibrate')
        recalibrate.layout.width = '100px'
    
        selectors = widgets.HBox([plot.children[0], recalibrate])
        
        def toggle_recalibrate(_):
            if recalibrate.value:
                self.calibration_layout.display='flex'
                recalibrate.description = 'Done'
                self._feature_importance_chart.layout.display='none'
            else:
                self.calibration_layout.display='none'
                recalibrate.description = 'Recalibrate'
                self._feature_importance_chart.layout.display='flex'
        
        recalibrate.observe(toggle_recalibrate, names='value')

        plot.children = (
            selectors,
            plot.children[1],
            plot.children[2]
        )

        return plot
    
    def _generate_calibration_box(self, X, y):
        #self._calibration_display

        self.y_val = y.copy()
        self.y_val_prob = self.model.predict_score(X)
        
        action = widgets.Button(description='apply')
        action.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
        action.layout = widgets.Layout(
        height='27px', width='100px', margin='0 0 0 100px')
                
        recal_max_depth = widgets.IntSlider(
            description='max_depth',
            min=0, max=30, step=1, value=self.model.params.max_depth,
            style = {'description_width': 'initial'}
        )
        
        recal_min_info_gain = widgets.FloatSlider(
            description='min_info_gain',
            min=0.0001, max=0.2, step=0.0001,
            value=self.model.params.min_info_gain, readout_format='.4f',
            style = {'description_width': 'initial'}
        )
        
        recal_min_leaf_size = widgets.FloatSlider(
            description='min_leaf_size',
            min=0.0001, max=0.2, step=0.0001,
            value=self.model.params.min_leaf_size, readout_format='.4f',
            style = {'description_width': 'initial'}
        )
        
        recal_weight = widgets.FloatSlider(
            description='weight',
            style={'description_width': 'initial'},
            value=self.model.params.weight, min=0.001, max=10, step=0.01)
        
        recal_power_degree = widgets.IntSlider(
            description='power_degree',
            style={'description_width': 'initial'},
            value=self.model.params.power_degree, min=1, max=7, step=2)
        
        recal_sigmoid_exponent = widgets.FloatSlider(
            description='sigmoid_exponent',
            style={'description_width': 'initial'},
            value=self.model.params.sigmoid_exponent, min=0, max=6, step=0.25)
        
        divider = widgets.HTML(f'<hr class="solid">')
        
        _title = widgets.HTML('<h4>Recalibrate Parameters</h4>')

        def plot_flex_activation():
    
            @widgets.interactive
            def _activation(
                weight=recal_weight, power_degree=recal_power_degree,
                sigmoid_exponent=recal_sigmoid_exponent):

                freq = np.arange(0, 101, 1)
                _nums = [flex_activation(
                    i, weight, power_degree, sigmoid_exponent) for i in freq]

                data = pd.DataFrame({
                    "freq": freq,
                    "weight": _nums,
                })

                fig, ax = plt.subplots(figsize=(3, 2))

                ax1 = sns.lineplot(data=data, y='weight', x='freq')

                plt.show()
                
            w = _activation
                
            return w.children[-1]
                
        box = widgets.VBox([
            _title,
            recal_max_depth,
            recal_min_info_gain,
            recal_min_leaf_size,
            recal_weight,
            recal_power_degree,
            recal_sigmoid_exponent,
            plot_flex_activation(),
            divider,
            action
        ])
        
        def on_click(_):
            self.model.update_feature_params(
                features=[self.feature_selector.value],
                max_depth=recal_max_depth.value,
                min_info_gain=recal_min_info_gain.value,
                min_leaf_size=recal_min_leaf_size.value,
                weight=recal_weight.value,
                power_degree=recal_power_degree.value,
                sigmoid_exponent=recal_sigmoid_exponent.value
            )
            self.y_prob = self.model.predict_score(self.X)
            self.model._calibration_map = self.model._map_calibration(
                self.y, self.y_prob, self.smoother.value)
            
            self.profile_plot_data = self._generate_explain_plot_data()

            # update the prediction in place
            new_pred = self.model.predict_score(X)
            for i, v in enumerate(new_pred):
                self.y_val_prob[i] = v
            
            self._refresh()
            
            group = self._generate_feature_importance()
            self.feature_importance_bars.children = group.children
            self.__update_scenario = True

            # with self.scenarios:
            #     clear_output(wait=True)
            #     scenario = ScenarioClassification(self.model, self.ranges)
            #     display(scenario.run())
            
        action.on_click(on_click)
        
        return box
    
    def _refresh(self):
        self.threshold.value += 0.01
        self.threshold.value -= 0.01

        v = self.feature_selector.value
        self.feature_selector.value = None
        self.feature_selector.value = v
    
    def add_action_button(self, button):
        self.action_buttons.append(button)
        
    def profile(self, X, y):
        
        evaluation = widgets.Output()
        # self.scenarios = widgets.Output(
        #     layout = widgets.Layout(min_height='720px'))
        
        # with self.scenarios:
        #     print("Generating widgets...")
            
        divider = widgets.HTML(f'<hr class="solid">')
        
        fimp_title = widgets.HTML('<h4>Feature Importances</h4>')
        self.feature_importance_bars = self._generate_feature_importance()
        feature_importances = widgets.VBox([self.feature_importance_bars])

        feature_importances.layout = widgets.Layout(
            height = '390px', overflow_y='auto')
        
        self.calibration_box = self._generate_calibration_box(X, y)
        self.calibration_box.layout = self.calibration_layout

        self._feature_importance_chart = widgets.VBox(
            [fimp_title, feature_importances])
        
        left_columns = widgets.VBox(
            [self._feature_importance_chart, self.calibration_box])
        
        contribution_title = widgets.HTML('<h4>Contributions</h4>')
        contribution_plot = self._generate_contribution_plot()

        contribution_display = widgets.VBox(
            [contribution_title, contribution_plot])
        
        metrics_eval = self._evaluation_screen(X, y)
        
        with evaluation:
            #title = widgets.HTML('<h3>Model Profile</h3>')
            row0 = widgets.HBox([left_columns, contribution_display])
            
            accordion = widgets.Accordion([metrics_eval], selected_index=0)
            accordion.set_title(0, 'Evaluation')
            
            tab0 = widgets.VBox([row0, accordion])
            display(tab0)
            
        # Get numeric ranges from training set
        for c in self.model.numeric_columns:
            t = int if self.X[c].dtype == int else float
            rng = {
                'type': t,
                'lower': self.X[c].min(),
                'upper': self.X[c].max()
                }
            
            self.ranges[c] = rng
            
        #tabs = widgets.Tab([evaluation, self.scenarios])
        tabs = widgets.Tab([evaluation])
        tabs.set_title(0, 'Evaluation')
        #tabs.set_title(1, 'Scenario Analysis')
        
        close_button = widgets.Button(description='close')
        close_button.style = {
            "button_color": '#e14067',
            "text_color": 'white'
            }

        # with self.scenarios:
        #     clear_output(wait=True)
        #     scenario = ScenarioClassification(self.model, self.ranges)
        #     display(scenario.run())
        
        screen = widgets.VBox([
            tabs,
            divider
            ])
        
        return screen


class EvaluateRegressor:
    
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.initialise()

        self.y_val = None
        self.y_val_pred = None
        self.X_val = None

        self.action_buttons = []
        self.scenarios = None
        self.ranges = {}
        
    def initialise(self):
        
        self._feature_importance_chart = None
        
        self._calibration_display = None
        
        fimp = self.model.feature_importances
        cols = list(fimp.keys())
        cols.reverse()
        cols.append(None)
        
        self.feature_selector = widgets.Dropdown(
            options=cols)
        
        self.profile_plot_data = self._generate_explain_plot_data()
        self.feature_importance_bars = None
        
        self.kvt = KeyValueTable(
            {},
            table_width='300px',
            header_alignment='right',
            cell_alignment='center',
            padding='0 10px 0 0',
            header_color='white'
        )
        self.kvt.html_widget.layout.margin = '10px 0 0 0'
        
        self.calibration_layout = widgets.Layout(
            display='none',
            padding='20px',
            margin='0 0 0 50px',
            width = '400px'
        )

        def on_feature_change(_):
            
            f = self.feature_selector.value
            
            if f is None:
                return
            
            params = self.model._constructs[self.model.columns.index(f)].params
            
            self.calibration_box.children[1].value = params.max_depth
            self.calibration_box.children[2].value = params.min_info_gain
            self.calibration_box.children[3].value = params.min_leaf_size
            self.calibration_box.children[4].value = params.tail_sensitivity
            
        self.feature_selector.observe(on_feature_change)

        self.y_pred = self.model.predict(self.X)
                
    def _generate_explain_plot_data(self):
        return gen_x_plot_data(self.model, 3)
        
    def _generate_feature_importance(self):
        def explore(b):
            self.feature_selector.value = b.id

        fimp = self.model.feature_importances
        cols = list(fimp.keys())
        cols.reverse()

        group = BarGroup(items=cols)
        group.set_suffix(items=None, suffix='%')
        group.add_button(text='view', on_click=explore)
        group.displays_layout.flex = 'none'

        width = int(max([len(i) for i in self.model.columns]) * 7)

        group.label_layout.width = f'{width}px'

        for i, v in fimp.items():
            val = round(v*100, 2)
            group.set_value(i, val)

        return group.show()
    
    def _evaluation_screen(self, X, y):
        
        evaluation = self.model.evaluate(X, y)
        
        self.kvt.update_data(evaluation)

        # Define a colormap that smoothly transitions between the two colors
        cmap = LinearSegmentedColormap.from_list(
            'custom', ['#eeeeee', '#0080ea'], N=100, gamma=0.4)
        
        chart_type = widgets.ToggleButtons(
            options=['Comparison', 'Error Dist', 'Residuals'])

        def _plot(chart_type=chart_type):

            evaluation = self.model.evaluate(X, y)

            if chart_type == 'Comparison':
                
                sns.set(font_scale=1.5)
                custom_params = {
                    "axes.spines.right": False,
                    "axes.spines.top": False
                    }
                
                sns.set_theme(style="white", palette=None, rc=custom_params)
                
                fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
                
                y_pred = self.model.predict(X)

                plot_data = pd.DataFrame({
                    'predicted': y_pred,
                    'actual': y
                })

                ax1 = sns.scatterplot(
                    data=plot_data, x='actual', y='predicted', alpha=0.5,
                    color='#0080ea')

                limit = max([y.max(), y_pred.max()]) * 1.05

                # Set the plot title and labels
                ax1.set_xlim(0, limit)
                ax1.set_ylim(0, limit)

                diag_line, = ax.plot(
                    ax.get_xlim(), ax.get_ylim(), ls=":", c=".3")

                # Show the plot
                plt.show()
            
            elif chart_type == 'Error Dist':
                sns.set(font_scale=1.5)
                custom_params = {
                    "axes.spines.right": False,
                    "axes.spines.top": False
                    }
                
                sns.set_theme(style="white", palette=None, rc=custom_params)

                fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
                
                y_pred = self.model.predict(X)
                
                plot_data = pd.DataFrame({
                    'predicted': y_pred,
                    'actual': y
                })

                plot_data['error'] = plot_data['predicted'] - plot_data['actual']

                ax1 = sns.histplot(plot_data['error'], color='#0080ea')

                limit = max([y.max(), y_pred.max()]) * 1.05

                # Show the plot
                plt.show()
                
            else:
                sns.set(font_scale=1.5)
                custom_params = {
                    "axes.spines.right": False,
                    "axes.spines.top": False
                    }
                
                sns.set_theme(style="white", palette=None, rc=custom_params)

                fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

                y_pred = self.model.predict(X)

                plot_data = pd.DataFrame({
                    'predicted': y_pred,
                    'actual': y
                })

                plot_data['error'] = plot_data['predicted'] - plot_data['actual']

                ax1 = sns.scatterplot(
                    data=plot_data, x='predicted', y='error', alpha=0.5,
                    color='#e14067')
                
                limit = y_pred.max() * 1.05

                ax.hlines(y=0, xmin=0, xmax=limit)

                # Show the plot
                plt.show()

        chart_output = widgets.Output()
        
        chart = interactive(_plot)
        chart.children[0].description = ''

        with chart_output:
            display(widgets.HTML('<h4>Chart</h4>'))
            display(chart)
        
        bar_title = widgets.HTML('<h4>Performance</h4>')
        bar_columns = widgets.VBox([
            bar_title,
            self.kvt.html_widget
        ])
        
        bar_columns.layout = widgets.Layout(
            margin='8px 0 0 25px')
        
        screen = widgets.HBox([
            chart_output,
            bar_columns
        ])
        
        return screen
    
    def _generate_contribution_plot(self):
        
        cf1 = lambda x: '#e14067' if x < 0 else '#12b980'
        cf2 = lambda x: '#0080ea'
        
        
        plot_selector = widgets.ToggleButtons(options=['contribution', 'mean', 'frequency'])
        plot_selector.style.button_width = '102px'
        plot_selector.layout = widgets.Layout(margin='10px 0 0 90px')
        
        def _plot(feature=self.feature_selector, plot=plot_selector):
            custom_params = {
                "axes.spines.right": False,
                "axes.spines.top": False
                }
            
            sns.set_theme(style="ticks", rc=custom_params)
            
            if feature is None:
                return
            
            plot_data = self.profile_plot_data[
                self.profile_plot_data['feature'] == feature]

            if plot_data.empty:
                print('')
                return
            
            cf = cf1 if plot == 'contribution' else cf2
            cp = [cf(value) for value in plot_data[plot]]
            
            fig, ax = plt.subplots(figsize=(5.7,5.3))
            
            ax1 = sns.barplot(
                data=plot_data,
                x=plot,
                y='value',
                palette=cp
            )
            
            ax1.tick_params(axis='x', labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            ax1.set_xlabel(plot.title(), size=14)
            ax1.set_ylabel('Value / Range', size=14)
            
            if plot == 'mean':
                base = self.model.base_value
                plt.axvline(x=base, ls=':', color='#374151', lw=1)

            plt.title = 'Contributions'

            plt.show()
            
        plot = interactive(_plot)
        
        plot_selector.description = ''
        
        recalibrate = widgets.ToggleButton(description='Recalibrate')
        recalibrate.layout.width = '100px'
    
        selectors = widgets.HBox([plot.children[0], recalibrate])
        
        def toggle_recalibrate(_):
            if recalibrate.value:
                self.calibration_layout.display='flex'
                recalibrate.description = 'Done'
                self._feature_importance_chart.layout.display='none'
            else:
                self.calibration_layout.display='none'
                recalibrate.description = 'Recalibrate'
                self._feature_importance_chart.layout.display='flex'
        
        recalibrate.observe(toggle_recalibrate, names='value')

        plot.children = (
            selectors,
            plot.children[1],
            plot.children[2]
        )

        return plot
    
    def _generate_calibration_box(self, X, y):
        #self._calibration_display

        self.y_val = y.copy()
        self.y_val_pred = self.model.predict(X)
        
        # Apply button
        action = widgets.Button(description='apply')
        action.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
        action.layout = widgets.Layout(
        height='27px', width='100px', margin='0 0 0 100px')
        
        # Optimisation Click
        def on_opt_click():
            xnet = XEvolutionaryNetwork(self.model)
            xnet.fit(self.X, self.y, subset=[self.feature_selector.value])

            xnet.add_layer(Tighten(
                iterations=20, early_stopping=20, learning_rate=0.2))
            
            xnet.add_layer(
                Evolve(max_leaves=1, max_severity=0.05, generations=20,
                       early_stopping=10))
            
            xnet.add_layer(
                Tighten(iterations=20, early_stopping=20, learning_rate=0.05))
            
            callback = RegressionCallback(xnet)
            
            box.children = (box.children[0],) + (callback.group.show(),)
            xnet.optimise(callback=callback)
            box.children = (box.children[0],) + \
                (recal_max_depth,
                recal_min_info_gain,
                recal_min_leaf_size,
                recal_tail_sensitivity,
                divider,
                action,
                )
            
        recal_max_depth = widgets.IntSlider(
            description='max_depth',
            min=0, max=30, step=1, value=self.model.params.max_depth,
            style = {'description_width': 'initial'}
        )
        
        recal_min_info_gain = widgets.FloatSlider(
            description='min_info_gain',
            min=0.0001, max=0.2, step=0.0001,
            value=self.model.params.min_info_gain, readout_format='.4f',
            style = {'description_width': 'initial'}
        )
        
        recal_min_leaf_size = widgets.FloatSlider(
            description='min_leaf_size',
            min=0.0001, max=0.2, step=0.0001,
            value=self.model.params.min_leaf_size, readout_format='.4f',
            style = {'description_width': 'initial'}
        )
        
        recal_tail_sensitivity = widgets.FloatSlider(
            description='tail_sensitivity',
            style={'description_width': 'initial'},
            value=self.model.params.tail_sensitivity, min=1, max=2, step=0.01)
        
        divider = widgets.HTML(f'<hr class="solid">')
        
        _title = widgets.HTML('<h4>Recalibrate Parameters</h4>')
                
        box = widgets.VBox([
            _title,
            recal_max_depth,
            recal_min_info_gain,
            recal_min_leaf_size,
            recal_tail_sensitivity,
            divider,
            action
        ])
        
        def on_click(_):
  
            self.model.update_feature_params(
                features=[self.feature_selector.value],
                max_depth=recal_max_depth.value,
                min_info_gain=recal_min_info_gain.value,
                min_leaf_size=recal_min_leaf_size.value,
                weight=None,
                power_degree=None,
                sigmoid_exponent=None,
                tail_sensitivity=recal_tail_sensitivity.value
            )
            on_opt_click()
            #self.y_pred = self.model.predict(self.X)
            self.kvt.update_data(self.model.evaluate(self.X_val, self.y_val))

            self.profile_plot_data = self._generate_explain_plot_data()

            # update the prediction in place
            new_pred = self.model.predict(X)
            for i, v in enumerate(new_pred):
                self.y_val_pred[i] = v
            
            self._refresh()
            
            group = self._generate_feature_importance()
            self.feature_importance_bars.children = group.children
            self.__update_scenario = True

            # with self.scenarios:
            #     clear_output(wait=True)
            #     scenario = ScenarioRegression(self.model, self.ranges)
            #     display(scenario.run())
            
        action.on_click(on_click)
        
        return box
    
    def _refresh(self):

        v = self.feature_selector.value
        self.feature_selector.value = None
        self.feature_selector.value = v
    
    def add_action_button(self, button):
        self.action_buttons.append(button)
        
    def profile(self, X, y):
        
        self.X_val = X
        self.y_val = y
        
        evaluation = widgets.Output()
        # self.scenarios = widgets.Output(
        #     layout = widgets.Layout(min_height='720px'))
        # with self.scenarios:
        #     print("Generating widgets...")
            
        divider = widgets.HTML(f'<hr class="solid">')
        
        fimp_title = widgets.HTML('<h4>Feature Importances</h4>')
        self.feature_importance_bars = self._generate_feature_importance()
        feature_importances = widgets.VBox([self.feature_importance_bars])

        feature_importances.layout = widgets.Layout(
            height = '390px', overflow_y='auto')
        
        self.calibration_box = self._generate_calibration_box(X, y)
        self.calibration_box.layout = self.calibration_layout

        self._feature_importance_chart = widgets.VBox(
            [fimp_title, feature_importances])
        
        left_columns = widgets.VBox(
            [self._feature_importance_chart, self.calibration_box])
        
        contribution_title = widgets.HTML('<h4>Contributions</h4>')
        contribution_plot = self._generate_contribution_plot()

        contribution_display = widgets.VBox(
            [contribution_title, contribution_plot])
        
        metrics_eval = self._evaluation_screen(X, y)
        
        with evaluation:
            #title = widgets.HTML('<h3>Model Profile</h3>')
            row0 = widgets.HBox([left_columns, contribution_display])
            
            accordion = widgets.Accordion([metrics_eval], selected_index=0)
            accordion.set_title(0, 'Evaluation')
            
            tab0 = widgets.VBox([row0, accordion])
            display(tab0)
            
        # Get numeric ranges from training set
        for c in self.model.numeric_columns:
            t = int if self.X[c].dtype == int else float
            rng = {
                'type': t,
                'lower': self.X[c].min(),
                'upper': self.X[c].max()
                }
            
            self.ranges[c] = rng
            
        #tabs = widgets.Tab([evaluation, self.scenarios])
        tabs = widgets.Tab([evaluation])
        tabs.set_title(0, 'Evaluation')
        #tabs.set_title(1, 'Scenario Analysis')
        
        close_button = widgets.Button(description='close')
        close_button.style = {
            "button_color": '#e14067',
            "text_color": 'white'
            }

        # with self.scenarios:
        #     clear_output(wait=True)
        #     scenario = ScenarioRegression(self.model, self.ranges)
        #     display(scenario.run())
        
        screen = widgets.VBox([
            tabs,
            divider
            ])
        
        return screen
