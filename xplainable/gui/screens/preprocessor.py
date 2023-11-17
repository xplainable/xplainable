""" Copyright Xplainable Pty Ltd, 2023"""

import xplainable
from ...utils import *
from ...preprocessing import transformers as xtf
from ...quality import XScan
from ...preprocessing.pipeline import XPipeline
from ...utils.api import *

import pandas as pd
import inspect
from pandas.api.types import is_numeric_dtype, is_string_dtype
import pandas.api.types as pdtypes
import pickle


class Preprocessor:

    def __init__(self):
        
        pd.set_option('display.max_columns', 500)

        self.preprocessor_name = None
        self.description=None
        self.pipeline = XPipeline()
        self._df_trans = pd.DataFrame()
        self.state = 0
        self.scan = {}

        self.preprocessor_options = None
        self.screen = None

    def _scan_df(self, df):
        """ Scans dataframe for summary statistics

        Args:
            df (pandas.DataFrame): The dataframe to scan

        Returns:
            dict: summary statistics
        """

        scanner = XScan()
        scanner.scan(df)

        return scanner.profile

    def _scan_feature(self, ser):
        """ Scans series for summary statistics

        Args:
            df (pandas.Series): The series to scan

        Returns:
            dict: summary statistics
        """

        scanner = XScan()

        return scanner._scan_feature(ser)

    def save(self, filename):
        """ Saves serialised object locally

        Args:
            filename (str): The filepath to save to.
        """
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp)

    def transform(self, df):
        """ Applies pipeline transformers to dataframe

        Args:
            df (pandas.DataFrame): The dataframe to transform

        Returns:
            pandas.DataFrame: The transformed dataframe
        """
        return self.pipeline.transform(df)
    
    def preprocess(self, df):
        """ GUI for easily preprocessing data.

        Args:
            df (pandas.DataFrame): Dataframe to preprocess

        """

        try:
            from ..components import Header
            from .save import PreprocessorPersist
            from ipywidgets import interactive
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            import matplotlib.pyplot as plt
            import seaborn as sns
        
        except:
            raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[gui]' to use "
                          "this feature.") from None

        # // ---------- SUPPORTING FUNCTIONS ---------- //
        def reset_charts():
            """Forces chart value change to trigger refresh"""
            # Chart 1
            v = charts.children[0].children[0].value
            charts.children[0].children[0].value = None
            charts.children[0].children[0].value = v

            # Chart 2
            v = charts.children[1].children[0].children[0].value
            charts.children[1].children[0].children[0].value = None
            charts.children[1].children[0].children[0].value = v

        def refresh_dataset():
            """Updates the LiveDF dataframe"""
            df_display.df = self._df_trans
            data_block.children[0].value = False
            data_block.children[0].value = True

        def refresh_dropdown_options():
            """Updates dropdowns after dataset change"""
            cols = list(self._df_trans.columns)
            feature.options = [None]+cols
            dist_plot_feature.options = cols+[None]
            multi_plot_x.options = cols
            multi_plot_y.options = cols
            multi_plot_hue.options = [None]+cols

        def dataset_changed():
            """Updates elements on dataset change"""
            refresh_dataset()
            refresh_dropdown_options()

        def sync_feature_dropdowns_a(_):
            dist_plot_feature.value = feature.value

        def sync_feature_dropdowns_b(_):
            feature.value = dist_plot_feature.value

        def plot_distribution():
            """ Interative distribution plotting"""
            def _plot(Feature=dist_plot_feature):
                if Feature == None:
                    return
                ser = self._df_trans[Feature].dropna().copy()
                if len(ser) > 1000:
                   ser = ser.sample(1000)
                if pdtypes.is_string_dtype(ser):
                    ser = ser.apply(lambda x: str(x))
                    if ser.nunique() > 20:
                        print("Too many unique categories to plot")
                        return

                elif pdtypes.is_bool_dtype(ser):
                    ser = ser.astype(str).apply(lambda x: x.lower())
                sns.set(rc={'figure.figsize':(10,4)})
                sns.histplot(ser)
                plt.show()
            
            return _plot
        
        def plot_multiple():
            """ Plotting two features together"""
            def _plot(x=multi_plot_x, y=multi_plot_y, hue=multi_plot_hue):
                
                if x is None or y is None:
                    return
                samp = self._df_trans[[x, y]]
                if samp.shape[0] > 1000:
                    samp = samp.sample(1000)

                try:
                
                    if is_numeric_dtype(samp[x]) and is_numeric_dtype(samp[y]):
                        multi_plot_hue.layout.visibility = 'visible'
                        sns.set(rc={'figure.figsize':(8,5)})
                        sns.jointplot(data = samp, x=x, y=y, hue=hue)
                        plt.show()

                    elif is_string_dtype(samp[x]) and is_string_dtype(samp[y]):
                        if samp[x].nunique() > 20 or samp[y].nunique() > 20:
                            print("Too many unique categories to plot")
                            return
                        sns.set(rc={'figure.figsize':(8,5)})
                        multi_plot_hue.layout.visibility = 'hidden'
                        sns.histplot(data=samp, x=x, hue=y, multiple="stack")

                    else:
                        if pdtypes.is_string_dtype(samp[x]) and \
                            samp[x].nunique() > 20:

                            print("Too many unique categories to plot")
                            return

                        if pdtypes.is_string_dtype(samp[y]) and \
                            samp[y].nunique() > 20:

                            print("Too many unique categories to plot")
                            return

                        multi_plot_hue.layout.visibility = 'visible'
                        sns.set(rc={'figure.figsize':(8,5)})
                        sns.violinplot(data = samp, x=x, y=y, hue=hue)
                        plt.show()

                except:
                    print(f'Cannot plot {x} and {y}')
            
            widget = interactive(_plot)
            controls = widgets.HBox(
                widget.children[:-1],
                layout = widgets.Layout(flex_flow='row wrap', width='100%'))

            output = widget.children[-1]
            output.layout = widgets.Layout(margin='0 0 0 25px')
            return widgets.VBox([controls, output])

        def feature_dropdown_selected(_):
            """ Listens to feature selection"""
            if feature.value is not None:
                typ = "numeric" if is_numeric_dtype(
                    self._df_trans[feature.value]) else "categorical"

                single_feature_transformers.options = [""]+[c[0] for c in \
                    clsmembers if "supported_types" in c[1].__dict__ and typ in\
                         c[1].__dict__['supported_types']]

            clear_param_zone()
            single_feature_transformers.value = None

        def single_feature_tf_selected(_):
            """ Listens to single feature transformer selection"""
            if single_feature_transformers.value:
                t = [c[1] for c in clsmembers if \
                    c[0] == single_feature_transformers.value][0]()

                single_feature_transformers.transformer = t
                params = single_feature_transformers.transformer(
                    self._df_trans[feature.value])

                if params:
                    adder_params.children = params.children

                add_button.layout.visibility = 'visible'
                docs_text.value = get_tf_description(
                    single_feature_transformers.value)
            
            else:
                docs_text.value = ""

        def multi_feature_tf_selected(_):
            """ Listens to multi feature transformer selection"""
            if multi_feature_transformers.value:
                t = [c[1] for c in clsmembers if \
                    c[0] == multi_feature_transformers.value][0]()
                    
                multi_feature_transformers.transformer = t
                params = multi_feature_transformers.transformer(self._df_trans)
                
                if params:
                    adder_params.children = params.children

                add_button.layout.visibility = 'visible'
                docs_text.value = get_tf_description(
                    multi_feature_transformers.value)
            
            else:
                docs_text.value = ""

        def get_tf_description(name):
            """Retreives transformer function documentation (first line)"""
            matches = [c for c in clsmembers if c[0] == name]

            if len(matches) > 0:
                return matches[0][1].__doc__.split("\n\n")[0].strip()

            else:
                return "No documentation"

        def refresh_params():
            """ instantiates param values when no selection required"""
            if len(adder_params.children) > 0:
                try:
                    v = adder_params.children[0].value
                    adder_params.children[0].value = None
                    adder_params.children[0].value = v
                except:
                    pass

        def clear_param_zone():
            """Hides elements in the adder parameters sections"""
            adder_params.children = []
            add_button.layout.visibility = 'hidden'

        def clear_warning(b):
            err_display.clear_output()

        def generate_transformer_warning(e):

            warning_title = widgets.HTML(
                f'<h4 style="color: #e14067;">WARNING<h4>')
            
            message = widgets.HTML(f'<p>The transformer could not be added.<p>')
            error_message = widgets.HTML(f'<p>{str(e)}<p>')
            undo_button = widgets.Button(description='dismiss')
            undo_button.style = {
                "button_color": '#e14067',
                "text_color": 'white'
                }
            undo_button.on_click(clear_warning)

            box = widgets.VBox(
                [warning_title, message, error_message, undo_button])
            
            box.layout.margin = '0 0 0 20px'

            return box

        def add_button_clicked(_):
            """ This is applied when the add button is clicked"""
            refresh_params()
            if selector_tabs.selected_index == 0:
                v = single_feature_transformers.value
                new_transformer = single_feature_transformers.transformer
                tf_feature = feature.value
                self.pipeline.add_stages([{
                    "feature": feature.value,
                    "transformer": new_transformer
                }])
                single_feature_transformers.value = ""
            
            else:
                v = multi_feature_transformers.value
                new_transformer = multi_feature_transformers.transformer
                tf_feature = '__dataset__'
                self.pipeline.add_stages([{
                    "feature": "__dataset__",
                    "transformer": new_transformer
                }])
                multi_feature_transformers.value = ""

            head_before = self._df_trans.head(10)

            with err_display:
                try:
                    self._df_trans = self.pipeline.fit_transform(
                        self._df_trans, self.state)
                except Exception as e:
                    warn = generate_transformer_warning(e)
                    self.pipeline.stages = self.pipeline.stages[:-1]
                    if selector_tabs.selected_index == 0:
                        single_feature_transformers.value = v
                    else:
                        multi_feature_transformers.value = v
                    display(warn)
                    return

            head_after = self._df_trans.head(10)

            n = len(pipeline_list.options)
            tf_params = new_transformer.__dict__

            if n == 0:
                new_option = f'0: {tf_feature} --> {v} --> {tf_params}'

            elif pipeline_list.options[0] == "":
                new_option = f'{n - 1}: {tf_feature} --> {v} --> {tf_params}'
            else:
                new_option = f'{n}: {tf_feature} --> {v} --> {tf_params}'

            pipeline_list.options = (*pipeline_list.options, new_option)
            pipeline_list.options = (
                o for o in pipeline_list.options if o != "")

            dataset_changed()
            
            if tf_feature == '__dataset__':
                #rescan data
                if always_rescan_dataset.value:
                    scan_data(None)
            else:
                #rescan data
                if always_rescan_feature.value:
                    dist_plot_feature.value = tf_feature
                    scan_feature(None)

            clear_param_zone()
            self.state += 1

        def rerun_pipeline():
            """ Re-applies transformer pipeline and sets state"""
            self._df_trans = self.pipeline.fit_transform(df)
            self.state = len(self.pipeline.stages)

        # Drop selected stage
        def drop_button_clicked(_):
            """ This is applied when the drop button is clicked"""
            
            # Find the indicies of the dropped
            idx = pipeline_list.index
            pipeline_features = [
                self.pipeline.stages[i]['feature'] for i in idx]

            if not idx:
                return

            self.pipeline.stages = [
                v for i, v in enumerate(self.pipeline.stages) if i not in idx]

            pipeline_list.options = [
                f'{i}: {s["feature"]} --> {s["name"]} --> {s["transformer"].__dict__}' \
                    for i, s in enumerate(self.pipeline.stages)]

            rerun_pipeline()
            dataset_changed()

            #rescan data
            if always_rescan_dataset.value:
                scan_data(None)

            elif always_rescan_feature.value:
                update_scan_on_drop(pipeline_features)

            reset_charts()

        def reset_value():
            """ Updates Distplot value on data scan"""
            current_val = dist_plot_feature.value
            dist_plot_feature.value = None
            dist_plot_feature.value = current_val

        def update_report_tables():
            """ Updates all tabs in report table after scan"""
            profile_df = pd.DataFrame(self.scan).T
            with report_table_zone_1:
                clear_output()
                sub_df = profile_df[
                    profile_df['type'] == 'numeric'].dropna(axis=1, how='all')

                if sub_df.shape[0] == 0:
                    return
                
                sub_df = sub_df.drop(columns=['type'])
                display(sub_df)

            with report_table_zone_2:
                clear_output()
                sub_df = profile_df[profile_df['type'] == 'categorical'].dropna(
                    axis=1, how='all')

                if sub_df.shape[0] == 0:
                    return

                sub_df = sub_df.drop(columns=['type'])
                display(sub_df)

            with report_table_zone_3:
                clear_output()
                sub_df = profile_df[profile_df['type'] == 'nlp'].dropna(
                    axis=1, how='all')

                if sub_df.shape[0] == 0:
                    return

                sub_df = sub_df.drop(columns=['type'])
                display(sub_df)

        def scan_data(_):
            """ Scans entire dataset"""
            scan_all_button.description = 'Scanning...'
            scan_all_button.disabled = True
            report_table_tabs.layout.visibility = 'hidden'
            report_table_loading_zone.layout.visibility = 'visible'

            with report_table_loading_zone:
                clear_output()
                self.scan = self._scan_df(self._df_trans)

            report_table_loading_zone.layout.visibility = 'hidden'
            report_table_tabs.layout.visibility = 'visible'
            update_report_tables()
            
            scan_all_button.description = 'Re-scan'
            scan_all_button.disabled = False
            
            reset_value()

        def scan_feature(_):
            """ Scans a selected feature"""
            selected_feature = dist_plot_feature.value

            if selected_feature is None:
                return

            scan_feature_button.description = 'Scanning...'
            scan_feature_button.disabled = True
            self.scan[selected_feature] = self._scan_feature(
                self._df_trans[selected_feature])

            if scan_all_button.description != 'Scan':
                update_report_tables()

            scan_feature_button.description = 'Re-scan'
            scan_feature_button.disabled = False

            reset_value()

        def update_scan_on_drop(features):
            """ Scans a selected group of feature"""

            if len(features) is None:
                return

            scan_feature_button.description = 'Scanning...'
            scan_feature_button.disabled = True

            if '__dataset__' in features:
                scan_data(None)

            else:
                for f in features:
                    self.scan[f] = self._scan_feature(self._df_trans[f])

            if scan_all_button.description != 'Scan':
                update_report_tables()

            scan_feature_button.description = 'Re-scan'
            scan_feature_button.disabled = False

            reset_value()

        def change_progress_colour(v, t1, t2):
            """ Changes the colour of progress bars"""
            if v < t1:
                return 'success'
            elif v < t2:
                return 'warning'
            else:
                return 'danger'

        def get_skewness_summary(s):
            if abs(s) < 0.5:
                style= 'success'
                status = '(Not skewed)'

            elif abs(s) < 1:
                style = 'warning'
                status = '(Moderately skewed)'
            else:
                style = 'danger'
                status = '(Highly skewed)'
            
            return style, status

        def get_kurtosis_summary(s):
            if abs(s) < 3:
                style= 'success'
                status = '(Acceptable Kurtosis)'

            elif abs(s) < 10:
                style = 'warning'
                if s < 0:
                    status = '(Moderate negative kurtosis)'
                else:
                    status = '(Moderate positive kurtosis)'
            else:
                style = 'danger'
                if s < 0:
                    status = '(High negative kurtosis)'
                else:
                    status = '(High positive kurtosis)'
            
            return style, status

        def summary_report(_):
            """ Constructs the summary report tables"""
            if len(self.scan) == 0:
                return

            feature = dist_plot_feature.value

            def hide_output():
                for i in summary_metrics:
                    components[i].layout.display = 'none'

                report_table_output.layout.visibility = 'hidden'

            if feature is None:
                hide_output()
                return

            if feature not in self.scan:
                scan_feature_button.description = 'Scan'
                scan_all_button.description = 'Scan'
                feature_summary.layout.visibility = 'hidden'
                hide_output()
                return

            else:
                scan_feature_button.description = 'Re-scan'
                feature_summary.layout.visibility = 'visible'

            report_table_output.layout.visibility = 'visible'

            table = pd.DataFrame(
                {i: {feature: v} for i, v in self.scan[feature].items()}).T

            # Categorical Features
            if table.loc['type'].values[0] == 'categorical':
                metrics = [
                    'missing_pct', 'cardinality', 'mixed_case', 'mixed_type']

                for i in summary_metrics:
                    if i in metrics:
                        components[i].layout.display = 'flex'

                        val = round(self.scan[feature][i]*100, 2)
                        bar_widgets[i].value = val
                        label_widgets[i].value = f'{val}%'
                        bar_widgets[i].bar_style = change_progress_colour(
                            val, 5, 40)

                    else:
                        components[i].layout.display = 'none'

            # Numeric features
            elif table.loc['type'].values[0] == 'numeric':

                metrics = ['missing_pct', 'skewness', 'kurtosis']
                for i in summary_metrics:
                    if i == 'missing_pct':
                        components[i].layout.display = 'flex'
                        val = round(self.scan[feature][i]*100, 2)
                        bar_widgets[i].value = val
                        label_widgets[i].value = f'{val}%'
                        bar_widgets[i].bar_style = change_progress_colour(
                            val, 5, 40)
                    elif i == 'skewness':
                        components[i].layout.display = 'flex'
                        val = self.scan[feature][i]
                        bar_widgets[i].value = abs(val)

                        style, status = get_skewness_summary(val)
                        bar_widgets[i].bar_style = style
                        label_widgets[i].value = status

                    elif i == 'kurtosis':
                        components[i].layout.display = 'flex'
                        val = self.scan[feature][i]
                        bar_widgets[i].value = abs(val)

                        style, status = get_kurtosis_summary(val)
                        bar_widgets[i].bar_style = style
                        label_widgets[i].value = status

                    else:
                        components[i].layout.display = 'none'

            # NLP Feature
            elif table.loc['type'].values[0] == 'nlp':
                metrics = [
                    'missing_pct', 'mixed_case', 'mixed_type']

                for i in summary_metrics:
                    if i in metrics:
                        components[i].layout.display = 'flex'
                        val = round(self.scan[feature][i]*100, 2)
                        bar_widgets[i].value = val
                        label_widgets[i].value = f'{val}%'
                        bar_widgets[i].bar_style = change_progress_colour(
                            val, 5, 40)
                    else:
                        components[i].layout.display = 'none'

            # NLP Feature
            elif table.loc['type'].values[0] == 'id':
                metrics = ['missing_pct']

                for i in summary_metrics:
                    if i in metrics:
                        components[i].layout.display = 'flex'
                        val = round(self.scan[feature][i]*100, 2)
                        bar_widgets[i].value = val
                        label_widgets[i].value = f'{val}%'
                        bar_widgets[i].bar_style = change_progress_colour(
                            val, 5, 40)
                    else:
                        components[i].layout.display = 'none'

            else:
                hide_output()
                metrics = []

            with report_table_output:
                clear_output(wait=True)
                display(table)  

        # //---------------------------------//


        # //------- INITIALISATION -------//
        # Instantiate transformed dataframe
        if self._df_trans.empty:
            self._df_trans = df.copy()
            if len(self.pipeline.stages) > 0:
                self._df_trans = self.transform(self._df_trans)

        # Retrieve all transformers
        clsmembers = inspect.getmembers(xtf, inspect.isclass)

        # //---------------------------------//


        # //------- HEADER -------//
        header = Header(title='Preprocessor', logo_size=40, font_size=18)
        header.title = {'margin': '10px 15px 0 10px'}

        # //---------------------------------//


        # //------- BODY (SELECTOR) -------//
        # Select Transformer Title
        select_transformer_title = widgets.HTML(f"<h4>Transformer</h4>")

        selector_layout = widgets.Layout(
            max_width='180px'
            )
        
        # Instantiate Feature select dropdown
        feature = widgets.Dropdown(
            options=[None]+list(self._df_trans.columns),
            layout=selector_layout
            )
        feature.observe(sync_feature_dropdowns_a, names=['value'])
        feature.observe(feature_dropdown_selected, names=['value'])
        
        # Instantiate transformer selection
        single_feature_transformers = TransformerDropdown(
            layout=selector_layout)

        single_feature_transformers.observe(
            single_feature_tf_selected, names=['value'])

        single_feature_selector = widgets.VBox(
            [feature, single_feature_transformers])

        # Get transformer options for multi-features
        multi_ops = [c[0] for c in clsmembers if "supported_types" in \
                c[1].__dict__ and 'dataset' in c[1].__dict__['supported_types']]

        multi_feature_transformers = TransformerDropdown(
            options= [""]+multi_ops,
            layout=selector_layout)

        multi_feature_transformers.observe(
            multi_feature_tf_selected, names=['value'])

        multi_feature_selector = widgets.VBox([multi_feature_transformers])

        # Compile transformer selectors to Tabs
        selector_tabs = widgets.Tab(
            [single_feature_selector, multi_feature_selector])

        selector_tabs.layout = widgets.Layout(
            max_width='230px'
            )

        selector_tabs.set_title(0, 'Single Feature')
        selector_tabs.set_title(1, 'Multi Feature')

        docs_text = widgets.HTML("")

        docs_box = widgets.Box([docs_text])
        docs_box.layout = widgets.Layout(
            max_width='200px',
            max_height='250px',
            display='flex',
            flex_flow='column wrap',
            margin = '0 0 0 20px'
            )

        selector = widgets.VBox([
            select_transformer_title,
            selector_tabs,
            docs_box
            ])

        selector.layout = widgets.Layout(
            min_width='250px',
            align_items='center',
            border='1px solid #7287a8'
            )

        # //---------------------------------//


        # //------- BODY (ADDER) -------//
        # Adder Title
        adder_title = widgets.HBox([widgets.HTML(f"<h4>Parameters</h4>")])
        adder_title.layout = widgets.Layout(align_items='center', height='auto')

        # Instantiate Add button (gets called in functions)
        add_button = TransformerButton(
            description="Add", icon='plus',
            layout=widgets.Layout(margin=' 10px 0 10px 0'))

        add_button.style.button_color = '#12b980'
        add_button.on_click(add_button_clicked)
        add_button.layout.visibility = 'hidden'

        adder_params = widgets.VBox([])
        adder_params.layout = widgets.Layout(
            max_height='500px',
            align_items="center",
            #overflow='scroll hidden'
            )

        # Instantiate VBox
        adder = widgets.VBox([
            adder_title,
            adder_params,
            add_button
            ])

        adder.layout = widgets.Layout(
            width='330px',
            align_items="center",
            border='1px solid #7287a8',
            margin=' 0 25px 0 25px '
            )

        # //---------------------------------//


        # //------- BODY (PIPELINE) -------//
        # Pipeline Title
        pipeline_title = widgets.HTML(f"<h4>Pipeline</h4>")

        # Instantiate pipeline output depending on stages
        if len(self.pipeline.stages) > 0:
            pipeline_list = widgets.SelectMultiple(
                options=[f'{i}: {s["feature"]} --> {s["name"]} --> {s["transformer"].__dict__}' \
                    for i, s in enumerate(self.pipeline.stages)])

        else:
            pipeline_list = widgets.SelectMultiple(options=[""])

        pipeline_list.layout = widgets.Layout(
            width='270px',
            height='150px')

        # Instantiate Drop button
        drop_button = widgets.Button(
            description="Drop Stage(s)",
            icon='times',
            layout=widgets.Layout(margin=' 10px 0 10px 0')
            )
        drop_button.style = {
            "button_color": '#e14067',
            "text_color": 'white'
            }
        drop_button.on_click(drop_button_clicked)

        # Compile pipeline display
        pipeline_display = widgets.VBox(
            [pipeline_title, pipeline_list, drop_button])

        pipeline_display.layout = widgets.Layout(
            width='300px',
            align_items='center',
            border='1px solid #7287a8'
            )

       # //---------------------------------//

       # //------- COMPILE BODY -------//
        body = widgets.HBox([selector, adder, pipeline_display])
        body.layout = widgets.Layout(margin = ' 0 0 0 25px ')

        # //---------------------------------//
        

        # //------- OUTPUT (DATAFRAME) -------//
        # Instantiate live dataframe object
        df_display = LiveDF(self._df_trans)
        data_block = df_display()
        data_block.children[0].layout.visibility = 'hidden'
        data_block.layout = widgets.Layout(
            display='flex',
            flex_flow='column wrap',
            width='100%',
            height='100%',
            align_items='initial'
            )

        # //---------------------------------//

        # //------- OUTPUT (PLOTS) -------//

        # Instantiate plotting dropdowns
        dist_plot_feature = widgets.Dropdown(
            options = list(self._df_trans.columns)+[None])

        dist_plot_feature.observe(sync_feature_dropdowns_b, names=['value'])
        dist_plot_feature.observe(summary_report, names=['value'])

        multi_layout = widgets.Layout(width='250px')
        multi_plot_x = widgets.Dropdown(
            options = self._df_trans.columns, layout=multi_layout)
            
        multi_plot_y = widgets.Dropdown(
            options = self._df_trans.columns, layout=multi_layout)
        multi_plot_hue = widgets.Dropdown(
            options = [None]+list(self._df_trans.columns), layout=multi_layout)

        # Display charts
        visuals = widgets.Output()
        with visuals:
            chart_a = interactive(plot_distribution())
            chart_b = plot_multiple()
            charts = widgets.Tab([chart_a, chart_b])
            charts.set_title(0, 'Distribution')
            charts.set_title(1, 'Interaction')
            display(charts)

        # //---------------------------------//
        

        # //------- OUTPUT (SUMMARY STATS) -------//
        # Instantiate summary metrics
        summary_metrics = [
            'missing_pct',
            'cardinality',
            'mixed_case',
            'mixed_type',
            'skewness',
            'kurtosis'
            ]

        # Progress bars for scan visualisations
        bar_widgets = {
            'missing_pct': widgets.FloatProgress(
                description = "Missing: ", value=0, max=100),
            'cardinality': widgets.FloatProgress(
                description = "Cardinality: ", value=0, max=100),
            'mixed_case': widgets.FloatProgress(
                description = "Mixed Cases: ", value=0, max=100),
            'mixed_type': widgets.FloatProgress(
                description = "Mixed Types: ", value=0, max=100),
            'skewness': widgets.FloatProgress(
                description = "Skewness: ", value=0, max=1.5),
            'kurtosis': widgets.FloatProgress(
                description = "Kurtosis: ", value=0, max=10)
        }

        # Labels for progress bars
        label_widgets = {
            i: widgets.HTML(f"") for i in summary_metrics
        }

        # Join bars and labels
        components = {
            i: widgets.HBox(
                [bar_widgets[i], label_widgets[i]]) for i in summary_metrics
        }

        # Hide components until feature selected
        for i in summary_metrics:
            components[i].layout.display = 'none' 
            components[i].style = {'description_width': 'initial'}

        # Compile bars
        health_bars = widgets.VBox([
            widgets.HTML("<b>Overview</b>"),
            widgets.HTML('<hr class="solid">'),
            components['missing_pct'],
            components['mixed_case'],
            components['mixed_type'],
            components['cardinality'],
            components['skewness'],
            components['kurtosis']
        ])

        health_bars.layout = widgets.Layout(margin='30px 0 0 50px')

        # Instantiate report table output (for single feature)
        report_table_output = widgets.Output()
        report_table_box = widgets.VBox([
            widgets.HTML("<b>Health Statistics</b>"),
            report_table_output
        ])

        report_table_box.layout = widgets.Layout(margin='30px 0 0 0')

        # Join single feature report table and bars 
        feature_summary = widgets.HBox([report_table_box, health_bars])
        feature_summary.layout = widgets.Layout(
            margin='0 0 15px 15px',
            visibility='hidden'
            )

        # Set feature summary heading
        feature_summary_heading = widgets.HTML("<h4>Feature Summary<h4>")
        feature_summary_heading.layout = widgets.Layout(margin='0 0 0 25px')

        # Instantiate scan feature button (scans single feature)
        scan_feature_button = widgets.Button(description='Scan')
        scan_feature_button.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
        scan_feature_button.on_click(scan_feature)

        # If enabled, scans after ever single-feature transformer
        always_rescan_feature = widgets.ToggleButton(
            value=True, description='always re-scan')

        # Reuse dist_plot_feature for each syncing
        feature_summary_selector = widgets.HBox(
            [dist_plot_feature, scan_feature_button, always_rescan_feature])

        # Compile feature summary body
        feature_summary_body = widgets.VBox(
            [feature_summary_heading, feature_summary_selector, feature_summary])
        feature_summary_body.layout = widgets.Layout(min_width='520px')
        
        report_table_heading = widgets.HTML("<h4>Dataset Summary Report</h4>")

        scan_all_button = widgets.Button(description='Scan')
        scan_all_button.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
        scan_all_button.on_click(scan_data)
        
        # If enabled, scans after ever multi-feature transformer
        always_rescan_dataset = widgets.ToggleButton(
            value=False, description='always re-scan')

        # Build report table heading
        report_table_rescan_box = widgets.HBox(
            [scan_all_button, always_rescan_dataset])


        # Zone layout
        zone_layout = widgets.Layout(
                width='100%',
                height='100%'
            )

        # Instantiate output zone for report tables
        report_table_loading_zone = widgets.Output()
        report_table_zone_1 = widgets.Output(layout=zone_layout)
        report_table_zone_2 = widgets.Output(layout=zone_layout)
        report_table_zone_3 = widgets.Output(layout=zone_layout)

        # Construct report table tabs
        report_table_tabs = widgets.Tab(
            [report_table_zone_1, report_table_zone_2, report_table_zone_3])

        report_table_tabs.set_title(0, 'Numeric')
        report_table_tabs.set_title(1, 'Categorical')
        report_table_tabs.set_title(2, 'NLP')

        # Hide report tables on instantiation
        report_table_tabs.layout.visibility = 'hidden'

        # Build report table summary body
        report_summary = widgets.VBox([
            report_table_heading,
            report_table_rescan_box,
            report_table_loading_zone,
            report_table_tabs
            ])

        report_summary.layout = widgets.Layout(margin='0 0 0 25px')

        # Compile summary statistics output
        summary_body = widgets.Tab([feature_summary_body, report_summary])
        summary_body.set_title(0, 'Feature')
        summary_body.set_title(1, 'Dataset')

        summary_body.layout = widgets.Layout(
            height='450px'
            )

        # //---------------------------------//

        # //------- COMPILE OUTPUT TABS -------//
        display_tabs = widgets.Tab([data_block, visuals, summary_body])
        display_tabs.set_title(0, 'Data')
        display_tabs.set_title(1, 'Plots')
        display_tabs.set_title(2, 'Summary Statistics')
        display_tabs.layout = widgets.Layout(margin = ' 25px 25px 25px 25px ')
        # //---------------------------------//


        # //------- FOOTER -------//
        save = PreprocessorPersist(self)
        footer = save.save()

        # //---------------------------------//
        err_display = widgets.Output()

        # //------- COMPILE SCREEN -------//
        self.screen = widgets.VBox(
            [header.show(), body, err_display, display_tabs, footer])

        # //---------------------------------//
        
        # DISPLAY ALL
        display(self.screen)
