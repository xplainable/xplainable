
import pandas as pd
from ipywidgets import interactive
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
from xplainable.preprocessing import transformers as tf
from xplainable.quality import XScan
from xplainable.preprocessing.pipeline import XPipeline
from xplainable.utils import *
import inspect
from pandas.api.types import is_numeric_dtype, is_string_dtype
import pandas.api.types as pdtypes
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from xplainable.utils.api import *
import xplainable
import time
import dill

class Preprocessor:

    def __init__(self, preprocessor_name):
        
        pd.set_option('display.max_columns', 500)
        warnings.filterwarnings('ignore')

        self.preprocessor_name = preprocessor_name
        self.pipeline = XPipeline()
        self._df_trans = pd.DataFrame()
        self.state = 0
        self.scan = {}
        self.df_delta = []

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
            dill.dump(self, outp)

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
            reset_charts()

            if save_pp.disabled:
                save_pp.description = "Save Preprocessor"
                save_pp.disabled = False

            if save_df.disabled:
                save_df.description = "Save Dataframe"
                save_df.disabled = False

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
                #if len(ser) > 10000:
                #   ser = ser.sample(10000)
                if pdtypes.is_string_dtype(ser):
                    ser = ser.apply(lambda x: str(x))
                    if ser.nunique() > 20:
                        print("Too many unique categories to plot")
                        return

                elif pdtypes.is_bool_dtype(ser):
                    ser = ser.astype(str).lower()
                sns.set(rc={'figure.figsize':(8,5)})
                sns.histplot(ser)
                plt.show()
            
            return _plot
        
        def plot_multiple():
            """ Plotting two features together"""
            def _plot(x=multi_plot_x, y=multi_plot_y, hue=multi_plot_hue):
                
                if x is None or y is None:
                    return
                samp = self._df_trans[[x, y]]
                #if samp.shape[0] > 10000:
                 #   samp = samp.sample(10000)

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
                widget.children[:-1], layout = widgets.Layout(flex_flow='row'))

            output = widget.children[-1]
            output.layout = widgets.Layout(margin='0 0 0 50px')
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

        # Get the difference in dataframes from dataset transformers
        def get_df_delta(df1, df2):
            """ Gets the delta between two dataframs"""
            output = {
                "drop": [col for col in df1.columns if col not in df2.columns],
                "add": [{"feature": col, "values": df2[col].head(10).to_json(
                    orient='records')} for col in df2.columns if \
                        col not in df1.columns]
            }
            return output

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
            adder_params.children = []
            add_button.layout.visibility = 'hidden'

        def add_button_clicked(_):
            """ This is applied when the add button is clicked"""
            refresh_params()
            if selector_tabs.selected_index == 0:
                v = single_feature_transformers.value
                new_transformer = single_feature_transformers.transformer
                tf_feature = feature.value
                self.pipeline.add_stages([{
                    "feature": feature.value,
                    "transformer": new_transformer,
                    "name": v
                }])
                single_feature_transformers.value = ""
            
            else:
                v = multi_feature_transformers.value
                new_transformer = multi_feature_transformers.transformer
                tf_feature = '__dataset__'
                self.pipeline.add_stages([{
                    "feature": "__dataset__",
                    "transformer": new_transformer,
                    "name": v
                }])
                multi_feature_transformers.value = ""

            head_before = self._df_trans.head(10)
            self._df_trans = self.pipeline.fit_transform(
                self._df_trans, self.state)

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

            # Store delta change
            if tf_feature == '__dataset__':
                self.df_delta.append(
                    get_df_delta(head_before, head_after)
                    )
                #rescan data
                if always_rescan_dataset.value:
                    scan_data(None)
            else:
                self.df_delta.append({"update": {
                    "feature": tf_feature,
                    "values": head_after[tf_feature].to_json(orient='records')
                    }})

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
            idx = pipeline_list.index

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

        def close_button_clicked(_):
            """Clears all output on close"""
            clear_output()

        def save_preprocessor_df(_):
            """Saves current state dataframe"""
            save_df.description = "Saving..."
            ts = str(int(time.time()))
            
            self._df_trans.to_csv(
                f'{self.preprocessor_name}_{ts}.csv',
                index=False)

            save_df.description = "Saved"
            save_df.disabled = True

        def save_preprocessor(_):
            """ Saves preprocessor to cloud"""
            save_pp.description = "Saving..."
            metadata = []
            for stage in self.pipeline.stages:
                step = {
                    'feature': stage['feature'],
                    'name': stage['name'],
                    'params': stage['transformer'].__dict__
                }
                
                metadata.append(step)

            try:

                # Get user models
                response = xplainable.client.__session__.get(
                    f'{xplainable.__client__.api_hostname}/preprocessors'
                    )

                # Prepare params
                user_preprocessors = get_response_content(response)

                api_params = {"preprocessor_name": self.preprocessor_name}

                # Create model if model name doesn't exist
                if not any(m['preprocessor_name'] == self.preprocessor_name for \
                    m in user_preprocessors):
                    
                    # Create preprocessor
                    response = xplainable.client.__session__.post(
                        f'{xplainable.__client__.api_hostname}/create-preprocessor',
                        params=api_params
                        )

                    # Check response content and fetch id
                    preprocessor_id = get_response_content(response)

                else:

                    response = xplainable.client.__session__.get(
                        f'{xplainable.__client__.api_hostname}/get-preprocessor-id',
                        params=api_params
                    )

                    preprocessor_id = get_response_content(response)


                # Prepare params and insert data
                api_params = {"preprocessor_id": preprocessor_id}

                insert_data = {
                    "stages": metadata,
                    "deltas": self.df_delta
                }

                # Create preprocessor version
                response = xplainable.client.__session__.post(
                    f'{xplainable.__client__.api_hostname}/preprocessors/{preprocessor_id}/add-version',
                    params=api_params,
                    json=insert_data
                    )

                # Check response content and fetch id
                preprocessor_id = get_response_content(response)

                save_pp.description = "Saved"
                save_pp.disabled = True

            except Exception as e:
                print(e)
                save_pp.description = "Failed"
                save_pp.disabled = True
                time.sleep(2)
                save_pp.description = "Try again"
                save_pp.disabled = False

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
                    report_table_zone_1.layout.visibility = 'hidden'
                    return

                sub_df = sub_df.drop(columns=['type'])
                display(sub_df)

            with report_table_zone_2:
                clear_output()
                sub_df = profile_df[profile_df['type'] == 'categorical'].dropna(
                    axis=1, how='all')

                if sub_df.shape[0] == 0:
                    report_table_zone_2.layout.visibility = 'hidden'
                    return

                sub_df = sub_df.drop(columns=['type'])
                display(sub_df)

            with report_table_zone_3:
                clear_output()
                sub_df = profile_df[profile_df['type'] == 'nlp'].dropna(
                    axis=1, how='all')

                if sub_df.shape[0] == 0:
                    report_table_zone_3.layout.visibility = 'hidden'
                    return

                sub_df = sub_df.drop(columns=['type'])
                display(sub_df)

        def scan_data(_):
            """ Scans entire dataset"""
            scan_all_button.description = 'Scanning...'
            scan_all_button.disabled = True
            report_table_tabs.layout.visibility = 'hidden'
            report_table_loading_zone.layout.display = 'flex'

            with report_table_loading_zone:
                clear_output()
                self.scan = self._scan_df(self._df_trans)

            report_table_loading_zone.layout.display = 'none'
            report_table_tabs.layout.visibility = 'visible'
            update_report_tables()
            
            scan_all_button.description = 'Re-scan Data'
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

            if scan_all_button.description != 'Scan Dataset':
                update_report_tables()

            scan_feature_button.description = 'Re-scan Feature'
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

        def summary_report(_):
            """ Constructs the summary report tables"""
            if len(self.scan) == 0:
                return

            feature = dist_plot_feature.value

            def hide_output():
                metrics = [
                    'missing_pct', 'cardinality', 'mixed_case', 'mixed_type']

                for i in metrics:
                    components[i].layout.visibility = 'hidden'

                report_table_output.layout.visibility = 'hidden'

            if feature is None:
                hide_output()
                return

            if feature not in self.scan:
                scan_feature_button.description = 'Scan Feature'
                scan_all_button.description = 'Scan Dataset'
                hide_output()
                return

            else:
                scan_feature_button.description = 'Re-scan Feature'

            report_table_output.layout.visibility = 'visible'

            table = pd.DataFrame(
                {i: {feature: v} for i, v in self.scan[feature].items()}).T

            if table.loc['type'].values[0] == 'categorical':
                metrics = [
                    'missing_pct', 'cardinality', 'mixed_case', 'mixed_type']

                for i in metrics:
                    components[i].layout.visibility = 'visible'

            elif table.loc['type'].values[0] == 'numeric':

                for i in ['cardinality', 'mixed_case', 'mixed_type']:
                    components[i].layout.visibility = 'hidden'

                components['missing_pct'].layout.visibility = 'visible' 
                metrics = ['missing_pct']

            elif table.loc['type'].values[0] == 'nlp':
                components['cardinality'].layout.visibility = 'hidden'
                metrics = ['missing_pct', 'mixed_case', 'mixed_type']

                for i in metrics:
                    components[i].layout.visibility = 'visible'

            else:
                metrics = [
                    'missing_pct', 'cardinality', 'mixed_case', 'mixed_type']

                for i in metrics:
                    components[i].layout.visibility = 'hidden'

                metrics = []

            for metric in metrics:
                val = round(self.scan[feature][metric]*100,2)
                bar_widgets[metric].value = val
                label_widgets[metric].value = f'{val}%'
                bar_widgets[metric].bar_style = change_progress_colour(
                    val, 5, 40)  

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

        # Instantiate delta tracking
        if len(self.df_delta) == 0:
            self.df_delta.append(
                {"start": df.head(10).to_json(orient='records')})

        # Retrieve all transformers
        clsmembers = inspect.getmembers(tf, inspect.isclass)

        # //---------------------------------//


        # //------- HEADER -------//
        # Load logo
        logo = open('xplainable/_img/logo.png', 'rb').read()
        logo_display = widgets.Image(
            value=logo, format='png', width=50, height=50)
        
        # Load preprocessor image
        label = open('xplainable/_img/label_preprocessor.png', 'rb').read()
        label_display = widgets.Image(value=label, format='png')

        # Build Header
        header = widgets.HBox([logo_display, label_display])
        header.layout = widgets.Layout(margin = ' 5px 0 15px 25px ')

        # //---------------------------------//


        # //------- BODY (SELECTOR) -------//
        # Select Transformer Title
        select_transformer_title = widgets.HTML(f"<h4>Transformer</h4>")
        
        # Instantiate Feature select dropdown
        feature = widgets.Dropdown(options=[None]+list(self._df_trans.columns))
        feature.observe(sync_feature_dropdowns_a, names=['value'])
        feature.observe(feature_dropdown_selected, names=['value'])

        # Instantiate transformer selection
        single_feature_transformers = TransformerDropdown()
        single_feature_transformers.observe(
            single_feature_tf_selected, names=['value'])

        single_feature_selector = widgets.VBox(
            [feature, single_feature_transformers])

        multi_feature_transformers = TransformerDropdown(
            options= [""]+[c[0] for c in clsmembers if "supported_types" in \
                c[1].__dict__ and 'dataset' in c[1].__dict__['supported_types']]
                )

        multi_feature_transformers.observe(
            multi_feature_tf_selected, names=['value'])

        multi_feature_selector = widgets.VBox([multi_feature_transformers])

        # Compile transformer selectors to Tabs
        selector_tabs = widgets.Tab(
            [single_feature_selector, multi_feature_selector])

        selector_tabs.set_title(0, 'Single Feature')
        selector_tabs.set_title(1, 'Multi Feature')

        docs_text = widgets.HTML("")

        docs_box = widgets.Box([docs_text])
        docs_box.layout = widgets.Layout(
            max_width='330px',
            max_height='250px',
            display='flex',
            flex_flow='column wrap',
            margin = '0 0 0 15px'
            )

        selector = widgets.VBox([
            select_transformer_title,
            selector_tabs,
            docs_box
            ])

        selector.layout = widgets.Layout(
            min_width='350px',
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
        adder_params.layout = widgets.Layout(max_height='250px')

        # Instantiate VBox
        adder = widgets.VBox([
            adder_title,
            adder_params,
            add_button
            ])

        adder.layout = widgets.Layout(
            min_width='400px',
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

        pipeline_list.layout = widgets.Layout(min_width='250px', height='150px')

        # Instantiate Drop button
        drop_button = widgets.Button(
            description="Drop Stage(s)",
            icon='times',
            layout=widgets.Layout(margin=' 10px 0 10px 0'))

        drop_button.style.button_color = '#e14067'
        drop_button.on_click(drop_button_clicked)

        # Compile pipeline display
        pipeline_display = widgets.VBox(
            [pipeline_title, pipeline_list, drop_button])

        pipeline_display.layout = widgets.Layout(
            min_width='350px',
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

        multi_plot_x = widgets.Dropdown(options = self._df_trans.columns)
        multi_plot_y = widgets.Dropdown(options = self._df_trans.columns)
        multi_plot_hue = widgets.Dropdown(
            options = [None]+list(self._df_trans.columns))

        # Display charts
        visuals = widgets.Output()
        with visuals:
            chart_a = interactive(plot_distribution())
            chart_b = plot_multiple()
            charts = widgets.HBox([chart_a, chart_b])
            display(charts)

        # //---------------------------------//
        

        # //------- OUTPUT (SUMMARY STATS) -------//
        # Instantiate summary metrics
        summary_metrics = [
            'missing_pct', 'cardinality', 'mixed_case', 'mixed_type']

        # Progress bars for scan visualisations
        bar_widgets = {
            i: widgets.FloatProgress(
                description = f"{i}: ", value=0) for i in summary_metrics
        }

        # Labels for progress bars
        label_widgets = {
            i: widgets.HTML(f"0%") for i in summary_metrics
        }

        # Join bars and labels
        components = {
            i: widgets.HBox(
                [bar_widgets[i], label_widgets[i]]) for i in summary_metrics
        }

        # Hide components until feature selected
        for i in summary_metrics:
            components[i].layout.visibility = 'hidden' 

        # Compile bars
        health_bars = widgets.VBox([
            components['missing_pct'],
            components['mixed_case'],
            components['mixed_type'],
            components['cardinality']
        ])

        health_bars.layout = widgets.Layout(margin='0 0 0 25px')

        # Instantiate report table output (for single feature)
        report_table_output = widgets.Output()

        # Join single feature report table and bars 
        feature_summary = widgets.HBox([report_table_output, health_bars])
        feature_summary.layout = widgets.Layout(margin='0 0 15px 15px')

        # Set feature summary heading
        feature_summary_heading = widgets.HTML("<h3>Feature Summary<h3>")
        feature_summary_heading.layout = widgets.Layout(margin='0 0 0 25px')

        # Instantiate scan feature button (scans single feature)
        scan_feature_button = widgets.Button(description='Scan Feature')
        scan_feature_button.style.button_color = '#0080ea'
        scan_feature_button.on_click(scan_feature)

        # If enabled, scans after ever single-feature transformer
        always_rescan_feature = widgets.Checkbox(
            value=True, description='always rescan')
        always_rescan_feature.layout = widgets.Layout(margin='10px 0 0 -75px')

        # Reuse dist_plot_feature for each syncing
        sub_header = widgets.HBox(
            [dist_plot_feature, scan_feature_button, always_rescan_feature])

        # Compile feature summary body
        feature_summary_body = widgets.VBox(
            [sub_header, feature_summary_heading, feature_summary])
        feature_summary_body.layout = widgets.Layout(min_width='520px')
        
        report_table_heading = widgets.HTML("<h3>Dataset Summary Report</h3>")
        scan_all_button = widgets.Button(description='Scan Dataset')
        scan_all_button.style.button_color = '#0080ea'
        scan_all_button.on_click(scan_data)
        scan_all_button.layout = widgets.Layout(margin='17px 0 0 10px')
        
        # If enabled, scans after ever multi-feature transformer
        always_rescan_dataset = widgets.Checkbox(
            value=False, description='always rescan')

        always_rescan_dataset.layout = widgets.Layout(margin='23px 0 0 -75px')
        
        # Build report table heading
        report_table_heading_box = widgets.HBox(
            [report_table_heading, scan_all_button, always_rescan_dataset])

        # Instantiate output zone for report tables
        report_table_loading_zone = widgets.Output()
        report_table_zone_1 = widgets.Output()
        report_table_zone_2 = widgets.Output()
        report_table_zone_3 = widgets.Output()

        # Construct report table tabs
        report_table_tabs = widgets.Tab(
            [report_table_zone_1, report_table_zone_2, report_table_zone_3])

        report_table_tabs.set_title(0, 'Numeric')
        report_table_tabs.set_title(1, 'Categorical')
        report_table_tabs.set_title(2, 'NLP')

        report_table_tabs.layout = widgets.Layout(
            max_height='350px'
        )

        # Hide report tables on instantiation
        report_table_tabs.layout.visibility = 'hidden'

        # Update zone layouts for h-scrollings tables
        for zone in [
            report_table_zone_1, report_table_zone_2, report_table_zone_3]:

            zone.layout = widgets.Layout(
                display='flex',
                flex_flow='column wrap',
                width='100%',
                align_items='initial'
            )

        # Build report table summary body
        report_summary = widgets.VBox([
            report_table_heading_box,
            report_table_loading_zone,
            report_table_tabs
            ])

        report_summary.layout = widgets.Layout(
            margin='0 0 0 50px',
            width='100%',
            height='100%'
            )

        # Compile summary statistics output
        summary_body = widgets.HBox([feature_summary_body, report_summary])

        # //---------------------------------//

        # //------- COMPILE OUTPUT TABS -------//
        display_tabs = widgets.Tab([data_block, visuals, summary_body])
        display_tabs.set_title(0, 'Data')
        display_tabs.set_title(1, 'Plots')
        display_tabs.set_title(2, 'Summary Stats')
        display_tabs.layout = widgets.Layout(margin = ' 25px 25px 25px 25px ')
        # //---------------------------------//


        # //------- FOOTER -------//
        done = widgets.Button(description='Done')
        done.style.button_color = '#0080ea'
        done.layout = widgets.Layout(margin=' 0 0 10px 25px')
        done.on_click(close_button_clicked)

        save_pp = widgets.Button(description='Save Preprocessor', disabled=True)
        save_pp.style.button_color = '#12b980'
        save_pp.layout = widgets.Layout(margin=' 0 0 10px 10px')
        save_pp.on_click(save_preprocessor)

        save_df = widgets.Button(description='Save Dataframe', disabled=True)
        save_df.style.button_color = '#12b980'
        save_df.layout = widgets.Layout(margin=' 0 0 10px 10px')
        save_df.on_click(save_preprocessor_df)

        footer = widgets.HBox([done, save_pp, save_df])

        # //---------------------------------//


        # //------- COMPILE SCREEN -------//
        screen = widgets.VBox([header, body, display_tabs, footer])

        # //---------------------------------//
        
        # DISPLAY ALL
        display(screen)
