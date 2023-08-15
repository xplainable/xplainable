import pandas as pd
import numpy as np
from ...visualisation.explain import _generate_explain_plot_data
from ...utils.activation import flex_activation


class XFineTuner:
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.type_mapping = {
            'macro': 'macro avg',
            'weighted': 'weighted avg',
            'neg': '0',
            'pos': '1'
        }
        
        self.server = None
    
    def _plot_importances(self, feature):
        try:
            import altair as alt
        except ImportError:
            raise ImportError(
                "Optional dependencies not found. Please install "
                "them with `pip install xplainable[gui]' to use "
                "this feature."
            ) from None

        fi = pd.DataFrame(
                {i: {'importance': v} for i, v in \
                 self.model.feature_importances.items()}).T.reset_index()

        fi = fi.rename(columns={'index': 'feature'})
        fi['importance_label'] = fi['importance'].apply(
            lambda x: str(round(x*100, 1))+'%')

        brush = alt.selection_interval(encodings=['y'])

        feature_importances = alt.Chart(
            fi,
            title='Feature Importances').mark_bar(color='#0080ea').encode(
                x='importance:Q', y=alt.Y('feature:N',
                    sort=alt.SortField(field='-importance:Q')),
                tooltip='importance_label',
                color=alt.condition(
            alt.datum.feature == feature,
            alt.value('lightgray'), alt.value('#0080ea'))
        ).properties(width=165, height=300).transform_filter(
            brush)

        view = alt.Chart(fi).mark_bar(color='#0080ea').encode(
            y=alt.Y('feature:N',
                    sort=alt.SortField(field='-importance:Q'), axis=alt.Axis(
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
    
    def _plot_contributions(self, feature):
        try:
            import altair as alt
        except ImportError:
            raise ImportError(
                "Optional dependencies not found. Please install "
                "them with `pip install xplainable[gui]' to use "
                "this feature."
            ) from None
    
        if feature == '__all__':
            features = list(self.model.columns)
        else:
            features = [feature]

        plot_data = _generate_explain_plot_data(self.model)
        filt = plot_data[plot_data['feature'].isin(features)]
        
        dropdown = alt.selection_point(
            name='Select',
            fields=['column'],
            bind=alt.binding_select(
                options=['contribution', 'mean', 'frequency']),
            value='contribution'
        )

        brush = alt.selection_interval(encodings=['y'])

        profile = alt.Chart(filt, title='Contributions').mark_bar(
            color='#e14067').encode(
            x='val:Q',
            y=alt.Y('value',
                    sort=alt.SortField(field='index', order='descending')),
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
                width=165,
            height=300
            ).transform_filter(
                brush
            ).add_params(dropdown)


        view = alt.Chart(filt).mark_bar(color='#e14067').encode(
                y=alt.Y('value:N',
                        sort=alt.SortField(field='index:Q', order='ascending'),
                        axis=alt.Axis(labels=False, title=None)),
                x=alt.X('contribution:Q',
                        axis=alt.Axis(labels=False, title=None)),
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
    
    def _create_confusion_matrix(self, data):
        try:
            import altair as alt
        except ImportError:
            raise ImportError(
                "Optional dependencies not found. Please install "
                "them with `pip install xplainable[gui]' to use "
                "this feature."
            ) from None
        
        # Create dataframe
        df = pd.DataFrame(
            data,
            columns=['predicted_0', 'predicted_1'],
            index=['actual_0', 'actual_1']).reset_index()

        df = df.melt(
            id_vars='index',
            value_vars=['predicted_0', 'predicted_1'],
            var_name='Predicted', value_name='Count')

        # Map to actual and predicted labels
        df['Actual'] = df['index'].map({'actual_0': 0, 'actual_1': 1})
        df['Predicted'] = df['Predicted'].map(
            {'predicted_0': 0, 'predicted_1': 1})

        # Add column for total count per actual value for calculating percentage
        total_counts = df.groupby('Actual')['Count'].sum().to_dict()
        df['Total'] = df['Actual'].map(total_counts)

        # Calculate percentage value
        df['Percentage'] = df['Count'] / df['Total']

        # Add classification type
        def classify_type(row):
            if row['Actual'] == row['Predicted']:
                return 'TP' if row['Actual'] == 1 else 'TN'
            else:
                return 'FP' if row['Predicted'] == 1 else 'FN'
        df['Type'] = df.apply(classify_type, axis=1)

        # Create altair chart
        base = alt.Chart(df).encode(
            x=alt.X('Predicted:O', axis=None),
            y=alt.Y('Actual:O', axis=None),
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color('Count:Q', legend=None)
        )

        text = base.mark_text(baseline='middle', fontSize=20).encode(
            text='Count:Q',
            color=alt.condition(
                alt.datum.Count < (df.Count.max() / 2),
                alt.value('black'),
                alt.value('white')
            )
        )

        text_labels = base.mark_text(
            baseline='middle', fontSize=15, dx=60, dy=40).encode(
            text='Type:N',
            color=alt.condition(
                alt.datum.Count < (df.Count.max() / 2),
                alt.value('black'),
                alt.value('white')
            )
        )

        chart = (heatmap + text + text_labels).properties(
            width=300, height=200).configure_view(
            strokeWidth=0,  # removes the chart's border
            fill='#1f2937'  # set the fill color
        ).configure(background='#1f2937')  # set the background color

        # Render the chart
        return chart
    
    def _plot_calibration(self):
        try:
            import altair as alt
        except ImportError:
            raise ImportError(
                "Optional dependencies not found. Please install "
                "them with `pip install xplainable[gui]' to use "
                "this feature."
            ) from None

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(
            list(self.model._calibration_map.items()),
            columns=['score', 'probability'])

        # Base chart
        calibration_line = alt.Chart(df).mark_line().encode(
            x=alt.X('score:Q', scale=alt.Scale(domain=(0, 100)), title=None),
            y=alt.Y('probability:Q', scale=alt.Scale(domain=(0, 1)), title=None)
        )

        # dashed line
        red_line = alt.Chart(
            pd.DataFrame({'probability': [self.model.base_value]})).mark_rule(
            color='gray', strokeDash=[5, 5]).encode(y='probability:Q')

        # Gray diagonal line
        diag_data = pd.DataFrame({
            'x': [0, 100],
            'y': [0, 1]
        })
        diag_line = alt.Chart(diag_data).mark_line(
            color='gray', strokeDash=[1, 1, 5, 5]).encode(
            x='x:Q', y='y:Q')

        chart = (calibration_line + red_line + diag_line).properties(
            width=300, height=200).configure_view(
            strokeWidth=0,  # removes the chart's border
            fill='#1f2937'  # set the fill color
        ).configure(background='#1f2937').configure_axis(
            gridColor='#404040',  # faint grid lines
            domain=False,  # remove axis domain line
            tickColor='white',
            labelColor='white',
            titleColor='white'
        )

        return chart
    
    def _update_feature_params(
        self,
        feature,
        max_depth,
        min_leaf_size,
        min_info_gain,
        weight,
        power_degree,
        sigmoid_exponent,
        cr_type
    ):
        if feature == '__all__':
            features = list(self.model.columns)
        else:
            features = [feature]

        self.model.update_feature_params(
            features=features,
            max_depth = max_depth,
            min_leaf_size = min_leaf_size,
            min_info_gain = min_info_gain,
            weight = weight,
            power_degree = power_degree,
            sigmoid_exponent = sigmoid_exponent
        )

        report, cm, cr = self._update_evaluation()
        report.update(cr[self.type_mapping[cr_type]])
                
        if self.model.target_map:
            self.model._calibration_map = self.model._map_calibration(
                self.y_train.map(self.model.target_map),
                self.model.predict_score(self.X_train)
            )
        else:
            self.model._calibration_map = self.model._map_calibration(
                self.y_train,
                self.model.predict_score(self.X_train)
            )

        return (
            self._plot_importances(feature),
            self._plot_contributions(feature),
            self._create_confusion_matrix(cm),
            report,
            self._plot_calibration()
        )

    def _update_params(self, feature):
            if feature == '__all__':
                feature = self.model.columns[0]

            max_depth = self.model.feature_params[feature]['max_depth']
            min_leaf_size = self.model.feature_params[feature]['min_leaf_size']
            min_info_gain = self.model.feature_params[feature]['min_info_gain']
            weight = self.model.feature_params[feature]['weight']
            power_degree = self.model.feature_params[feature]['power_degree']
            sigmoid_exponent = self.model.feature_params[
                feature]['sigmoid_exponent']

            return (
                int(max_depth),
                float(min_leaf_size),
                float(min_info_gain),
                float(weight),
                int(power_degree),
                float(sigmoid_exponent)
            )

    def _update_plot(self, weight, power_degree, sigmoid_exponent):

        freq = np.arange(0, 101, 1)

        return pd.DataFrame({
            "freq": freq,
            "value": [flex_activation(
            i, weight, power_degree, sigmoid_exponent) for i in freq]})

    def _update_evaluation(self):

        report = self.model.evaluate(self.X_test, self.y_test)
        cm = report.pop('confusion_matrix')
        cr = report.pop('classification_report')
        acc = cr.pop('accuracy')
        for i in list(cr.keys()):
            cr[i].pop('support')

        report['log_loss'] = report['log_loss'] / 100

        return report, cm, cr

    def _update_cr_type(self, cr_type):
        report, cm, cr = self._update_evaluation()
        report.update(cr[self.type_mapping[cr_type]])
        
        return report
    
    def _run_server(self, height=1000, server_port=7860, share=False, auth=None):
        try:
            import gradio as gr
        except ImportError:
            raise ImportError(
                "Optional dependencies not found. Please install "
                "them with `pip install xplainable[gui]' to use "
                "this feature."
            ) from None

        with gr.Blocks() as self.server:

            starting = list(self.model.feature_importances.keys())[-1]
            p = self.model.feature_params[starting]

            freq = np.arange(0, 101, 1)
            _nums = pd.DataFrame({
                    "freq": freq,
                    "value": [flex_activation(i, 0.2, 1, 1) for i in freq]}
            )

            report, cm, cr = self._update_evaluation()
            report.update(cr['macro avg'])

            with gr.Tab(label='Tuning'):
                gr.Markdown("# Model Tuner")
                with gr.Row():
                    with gr.Column(scale=0):

                        features = gr.Dropdown(
                            choices=["__all__"]+self.model.columns,
                            value=starting, label="Features", interactive=True)
                        
                        btn = gr.Button("Refit Parameters")

                        max_depth = gr.Slider(
                            value=p['max_depth'], label="Max Depth", minimum=0,
                            maximum=12, step=1)
                        
                        min_leaf_size = gr.Slider(
                            value=p['min_leaf_size'], label="Min Leaf Size",
                            minimum=0, maximum=0.25, step=0.0005)
                        
                        min_info_gain = gr.Slider(
                            value=p['min_info_gain'], label="Min Info Gain",
                            minimum=0, maximum=0.25, step=0.0005)
                        
                        weight = gr.Slider(
                            value=p['weight'], label="Weight", minimum=0,
                            maximum=3, step=0.01)
                        
                        power_degree = gr.Slider(
                            value=p['power_degree'], label="Power Degree",
                            minimum=1, maximum=5, step=2)
                        
                        sigmoid_exponent = gr.Slider(
                            value=p['sigmoid_exponent'],
                            label="Sigmoid Exponent", minimum=0, maximum=1,
                            step=0.1)
                        
                        line_plot = gr.LinePlot(
                            _nums, x='freq', y='value', interactive=False)

                    with gr.Row():

                        with gr.Column():

                            with gr.Tab(label='Feature Importances'):
                                fi_outt = gr.Plot(
                                    label="",
                                    value=self._plot_importances(starting),
                                    visible=True)

                            with gr.Tab(label='Contributions'):
                                outt = gr.Plot(
                                    label="",
                                    value=self._plot_contributions(starting))

                        with gr.Column():

                            with gr.Tab(label='Confusion Matrix'):
                                cm_outt = gr.Plot(
                                    label="",
                                    value=self._create_confusion_matrix(cm),
                                    visible=True)

                            with gr.Tab(label='Calibration'):
                                cal_outt = gr.Plot(
                                    label="",
                                    value=self._plot_calibration(),
                                    visible=True)

                            with gr.Tab(label='Metrics'):
                                cr_type = gr.Radio(
                                    choices=['macro', 'weighted', 'neg', 'pos'],
                                    value='macro', label='Type')
                                metric_outt = gr.Label(report, label="Metrics")

            features.change(
                fn=self._update_params,
                inputs=[features],
                outputs=[
                    max_depth, min_leaf_size, min_info_gain, weight,
                    power_degree, sigmoid_exponent]
            )

            features.change(
                fn=self._plot_contributions,
                inputs=[features],
                outputs=[outt])

            features.change(
                fn=self._plot_importances,
                inputs=[features],
                outputs=[fi_outt])

            weight.change(
                fn=self._update_plot,
                inputs=[weight, power_degree, sigmoid_exponent],
                outputs=[line_plot])

            power_degree.change(
                fn=self._update_plot,
                inputs=[weight, power_degree, sigmoid_exponent],
                outputs=[line_plot])

            sigmoid_exponent.change(
                fn=self._update_plot,
                inputs=[weight, power_degree, sigmoid_exponent],
                outputs=[line_plot])

            cr_type.change(
                fn=self._update_cr_type,
                inputs=[cr_type],
                outputs=[metric_outt])

            btn.click(
                self._update_feature_params,
                [features, max_depth, min_leaf_size, min_info_gain, weight, \
                 power_degree, sigmoid_exponent, cr_type],
                [fi_outt, outt, cm_outt, metric_outt, cal_outt]
            )

        self.server.launch(
            inbrowser=False, inline=True, quiet=True, height=height,
            show_api=False, show_tips=False, show_error=True, share=share,
            auth=auth, server_port=server_port)
        
    def start(
        self, height=1000, server_port=7860, share=False, auth=None, \
            close=False):
        
        try:
            self._run_server(height, server_port, share, auth)
            
        except OSError as e:
            if close:
                self.server.close()
                print("Closed port and restarting server")
                self._run_server(height, server_port, share, auth)
            else:
                raise OSError(e)
            
    def stop(self):
        self.server.close()
