
import pandas as pd
from time import sleep
from ipywidgets import interact, interactive
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
from xplainable.preprocessing import transformers as tf
from xplainable.quality import XScan
from xplainable.preprocessing.pipeline import XPipeline
import inspect
from pandas.api.types import is_numeric_dtype, is_string_dtype
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from xplainable.utils import *

pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')


class DFViz:

    def __init__(self, df):
        self.df = df

    def __call__(self):
        
        def _set_params(viz=True):
            if viz:
                display(self.df)

        return interactive(_set_params)

class Preprocessor:

    def __init__(self, scan=True):
        self.pipeline = XPipeline()
        self._df_trans = pd.DataFrame()
        self.state = 0
        self.scan_result = {}
        self.scan = scan

    def _scan(self, df):
        scanner = XScan()
        scanner.scan(df)
        return scanner.profile

    def transform(self, df):
        return self.pipeline.transform(df)

    def preprocess(self, df):
        
        if self._df_trans.empty:
            self._df_trans = df.copy()

        # Run initial scan
        if self.scan:
            print("Scanning data for health statistics...")
            self.scan_result = self._scan(df)
            clear_output()

        clsmembers = inspect.getmembers(tf, inspect.isclass)
        
        # This allows widgets to show full label text
        style = {'description_width': 'initial'}

        feature = widgets.Dropdown(options = self._df_trans.columns)
        
        feature_transformers = TransformerDropdown()
        dataset_transformers = TransformerDropdown(options =  [""]+[c[0] for c in clsmembers if "supported_types" in c[1].__dict__ and 'dataset' in c[1].__dict__['supported_types']])
        params = widgets.VBox([])
        screen = widgets.HBox([])
        divider = widgets.HTML(
            f'<hr class="solid">', layout=widgets.Layout(height='auto'))

        # HEADER
        logo = open('xplainable/_img/logo.png', 'rb').read()
        logo_display = widgets.Image(
            value=logo, format='png', width=50, height=50)
        
        label = open('xplainable/_img/label_preprocessor.png', 'rb').read()
        label_display = widgets.Image(value=label, format='png')

        header = widgets.HBox([logo_display, label_display])
        header.layout = widgets.Layout(margin = ' 5px 0 15px 25px ')

        def reset_charts():
            v = charts.children[0].children[0].value
            charts.children[0].children[0].value = None
            #sleep(0.01)
            charts.children[0].children[0].value = v

        def dataset_changed():
            df_display.df = self._df_trans
            feature.options = list(self._df_trans.columns)
            chart_a_feature.options = list(self._df_trans.columns)+[None]
            chart_b_feature1.options = self._df_trans.columns
            chart_b_feature2.options = self._df_trans.columns
            chart_hue.options = [None]+list(self._df_trans.columns)
            
            w.children[0].value = False
            w.children[0].value = True
            reset_charts()
            
        chart_a_feature = widgets.Dropdown(options = list(self._df_trans.columns)+[None])
        def plot_distribution():
            def _plot(feature=chart_a_feature):
                if feature:
                    sns.set(rc={'figure.figsize':(8,5)})
                    sns.histplot(self._df_trans[feature])
                    plt.show()
            
            return _plot

        chart_b_feature1 = widgets.Dropdown(options = self._df_trans.columns)
        chart_b_feature2 = widgets.Dropdown(options = self._df_trans.columns)
        chart_hue = widgets.Dropdown(options = [None]+list(self._df_trans.columns))
        def plot_multiple():
            def _plot(x=chart_b_feature1, y=chart_b_feature2, hue=chart_hue):
                
                if is_numeric_dtype(self._df_trans[x]) and is_numeric_dtype(self._df_trans[y]):
                    chart_hue.layout.visibility = 'visible'
                    sns.set(rc={'figure.figsize':(8,5)})
                    sns.jointplot(data = self._df_trans, x=x, y=y, hue=hue)
                    plt.show()

                elif is_string_dtype(self._df_trans[x]) and is_string_dtype(self._df_trans[y]):
                    sns.set(rc={'figure.figsize':(8,5)})
                    chart_hue.layout.visibility = 'hidden'
                    sns.histplot(data=self._df_trans, x=x, hue=y, multiple="stack")

                else:
                    chart_hue.layout.visibility = 'visible'
                    sns.set(rc={'figure.figsize':(8,5)})
                    sns.violinplot(data = self._df_trans, x=x, y=y, hue=hue)
                    plt.show()
            
            widget = interactive(_plot)
            controls = widgets.HBox(widget.children[:-1], layout = widgets.Layout(flex_flow='row'))
            output = widget.children[-1]
            return widgets.VBox([controls, output])

        def selected_col(_):
            typ = "numeric" if is_numeric_dtype(self._df_trans[feature.value]) else "categorical"
            feature_transformers.options = [""]+[c[0] for c in clsmembers if "supported_types" in c[1].__dict__ and typ in c[1].__dict__['supported_types']]

        def feature_selected(d):
            if feature_transformers.value:
                t = [c[1] for c in clsmembers if c[0] == feature_transformers.value][0]()
                feature_transformers.transformer = t
                params = feature_transformers.transformer(df_display.df[feature.value])
                adder.children = transformer_title,
                if params:
                    adder.children += params.children
                
                adder.children += add_button,

            else:
                adder.children = transformer_title,
        
        def dataset_selected(d):
            if dataset_transformers.value:
                t = [c[1] for c in clsmembers if c[0] == dataset_transformers.value][0]()
                dataset_transformers.transformer = t
                params = dataset_transformers.transformer(df_display.df)
                
                adder.children = transformer_title,
                if params:
                    adder.children += params.children
        
                adder.children += add_button,

            else:
                adder.children = transformer_title,
                

        def add_button_clicked(_):
            
            if selector_tabs.selected_index == 0:
                v = feature_transformers.value
                new_transformer = feature_transformers.transformer
                tf_feature = feature.value
                self.pipeline.add_stages([{
                    "feature": feature.value,
                    "transformer": new_transformer,
                    "name": v
                }])
                feature_transformers.value = ""
            else:
                v = dataset_transformers.value
                new_transformer = dataset_transformers.transformer
                tf_feature = '__dataset__'
                self.pipeline.add_stages([{
                    "feature": "__dataset__",
                    "transformer": new_transformer,
                    "name": v
                }])
                dataset_transformers.value = ""

            self._df_trans = self.pipeline.fit_transform(self._df_trans, self.state)
            n = len(pipeline_list.options)
            tf_params = new_transformer.__dict__

            if n == 0:
                new_option = f'0: {tf_feature} --> {v} --> {tf_params}'

            elif pipeline_list.options[0] == "":
                new_option = f'{n - 1}: {tf_feature} --> {v} --> {tf_params}'
            else:
                new_option = f'{n}: {tf_feature} --> {v} --> {tf_params}'

            pipeline_list.options = (*pipeline_list.options, new_option)
            pipeline_list.options = (o for o in pipeline_list.options if o != "")

            dataset_changed()
        
            self.state += 1

        def rerun_pipeline():
            self._df_trans = self.pipeline.fit_transform(df)
            self.state = len(self.pipeline.stages)

        # Drop selected stage
        def drop_button_clicked(b):
            val = pipeline_list.value

            drop_indices = [int(i[0]) for i in val]
            self.pipeline.stages = [
                v for i, v in enumerate(self.pipeline.stages) if i not in drop_indices]

            if not val:
                return

            elif len(pipeline_list.options) == 1:
                pipeline_list.options = ('')

            elif val and val[0] != '':    
                pipeline_list.options = (o for o in pipeline_list.options if o not in val)

                pipeline_list.options = (
                    f'{i}:{o.split(":")[1]}' for i, o in enumerate(pipeline_list.options))

            rerun_pipeline()
            dataset_changed()

        def close_button_clicked(_):
            clear_output()

        feature.observe(selected_col, names=['value'])
        feature_transformers.observe(feature_selected, names=['value'])
        dataset_transformers.observe(dataset_selected, names=['value'])

        add_button = TransformerButton(description="Add", icon='plus')
        add_button.style.button_color = '#12b980'
        add_button.on_click(add_button_clicked)
        add_button.layout = widgets.Layout(margin=' 10px 0 10px 0')

            # Create drop button
        drop_button = widgets.Button(description="Drop Stage(s)",icon='times')
        drop_button.style.button_color = '#e14067'
        drop_button.layout = widgets.Layout(margin=' 10px 0 10px 0')
        drop_button.on_click(drop_button_clicked)

        feature_title = widgets.HTML(
            f"<h4>Transformer</h4>", layout=widgets.Layout(height='auto'))

        feature_selector = widgets.VBox([feature, feature_transformers])
        dataset_selector = widgets.VBox([dataset_transformers])

        selector_tabs = widgets.Tab([feature_selector, dataset_selector])
        selector_tabs.set_title(0, 'Features')
        selector_tabs.set_title(1, 'Dataset')
        selector = widgets.VBox([feature_title, selector_tabs])
        selector.layout = widgets.Layout(min_width='350px', align_items='center', border='1px solid #7287a8')

        transformer_title = widgets.HBox([widgets.HTML(f"<h4>Parameters</h4>")])
        transformer_title.layout = widgets.Layout(align_items='center', height='auto')

        adder = widgets.VBox([transformer_title, params])
        adder.layout = widgets.Layout(
            min_width='350px',
            align_items="center",
            border='1px solid #7287a8',
            margin = ' 0 25px 0 25px '
            )

        pipeline_title = widgets.HTML(
            f"<h4>Pipeline</h4>", layout=widgets.Layout(height='auto'))

        if len(self.pipeline.stages) > 0:
            pipeline_list = widgets.SelectMultiple(options=[f'{i}: {s["name"]}' for i, s in enumerate(self.pipeline.stages)])
            #df_display = DFViz(self.pipeline.fit_transform(df))

        else:
            pipeline_list = widgets.SelectMultiple(options=[""])

        pipeline_list.layout = widgets.Layout(min_width='250px', height='150px')
        
        df_display = DFViz(self._df_trans)

        pipeline_display = widgets.VBox([pipeline_title, pipeline_list, drop_button])
        pipeline_display.layout = widgets.Layout(min_width='350px', align_items='center', border='1px solid #7287a8')

        body = widgets.HBox([selector, adder, pipeline_display])
        body.layout = widgets.Layout(margin = ' 0 0 0 25px ')

        tabs = widgets.Box([body])

        done = widgets.Button(description='Done')
        done.style.button_color = '#0080ea'
        done.layout = widgets.Layout(margin=' 0 0 10px 25px')
        done.on_click(close_button_clicked)

        save_pp = widgets.Button(description='Save Preprocessor')
        save_pp.style.button_color = '#12b980'
        save_pp.layout = widgets.Layout(margin=' 0 0 10px 10px')
        save_pp.on_click(close_button_clicked)

        save_df = widgets.Button(description='Save Dataframe')
        save_df.style.button_color = '#12b980'
        save_df.layout = widgets.Layout(margin=' 0 0 10px 10px')
        save_df.on_click(close_button_clicked)

        footer = widgets.HBox([done, save_pp, save_df])

        # Data
        w = df_display()
        w.children[0].layout.visibility = 'hidden'
        data_block = widgets.HBox([w])
        w.layout = widgets.Layout(
                            display='flex',
                            flex_flow='column wrap',
                            width='100%',
                            height='100%',
                            align_items='initial'
                            )

        # Charts
        visuals = widgets.Output()
        display_tabs = widgets.Tab([data_block, visuals])
        display_tabs.set_title(0, 'Data')
        display_tabs.set_title(1, 'Visualise')
        display_tabs.layout = widgets.Layout(margin = ' 25px 25px 25px 25px ')
        screen = widgets.VBox([header, tabs, display_tabs, footer])

        with visuals:
            chart_a = interactive(plot_distribution())
            chart_b = plot_multiple()
            charts = widgets.HBox([chart_a, chart_b])
            display(charts)
            
        display(screen)
