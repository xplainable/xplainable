""" Copyright Xplainable Pty Ltd, 2023"""

from ...core.models import PartitionedRegressor, XRegressor
from ...core.optimisation.genetic import XEvolutionaryNetwork
from ...core.optimisation.layers import Tighten, Evolve
from ...quality import XScan
from ...gui.components.tables import KeyValueTable 
from ..components import Header
from ...utils.xwidgets import TrainButton
from ...callbacks.optimisation import RegressionCallback
from ...gui.components.bars import IncrementalBarPlot
from ...gui.components.pipelines import VisualPipeline
from .evaluate import EvaluateRegressor
from .save import ModelPersist
from ..components import BarGroup
from ...metrics.metrics import evaluate_regression
from ...utils.handlers import check_df
from ..._dependencies import _check_critical_versions


from sklearn.model_selection import train_test_split
import numpy as np
import copy
import time
import ipywidgets as widgets
from IPython.display import  display, clear_output
from sklearn.metrics import *

def regressor(df):
    """ Trains an xplainable regressor via a simple GUI.

    Args:
        df (pandas.DataFrame): Training data including target

    Raises:
        RuntimeError: When model fails to fit

    Returns:
        xplainale.models.XRegressor: The trained model
    """

    # Check critical dependencies
    _check_critical_versions()
    
    # Assert dataframe is valid
    check_df(df)

    style = {'description_width': 'initial'}
    divider = widgets.HTML(f'<hr class="solid">')
    
    header = Header(title='Train Regressor', logo_size=40, font_size=18)
    header.title = {'margin': '10px 15px 0 10px'}
    
    part_progress = BarGroup(items=['Partitions'])
    part_progress.collapse_items(items=['Partitions'])
    part_progress.window_layout.margin = '10px 0 0 0'

    header.add_widget(part_progress.show())

    result_out = widgets.Output()
    
    column_names = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    divider = widgets.HTML(f'<hr class="solid">')

    # COLUMN 1
    _title_target = widgets.HTML(
        f"<h5>Target</h5>", layout=widgets.Layout(height='auto'))
    
    _title_target.layout = widgets.Layout(margin='0 0 0 15px')

    _title_id = widgets.HTML(f"<h5>ID Column (0 selected)</h5>")
    
    _dropdown_target = widgets.Dropdown(
        options=numeric_columns,
        layout = widgets.Layout(width='200px')
        )

    possible_partitions = [None]+[i for i in df.columns if df[i].nunique() < 20]

    _dropdown_partition_on = widgets.Dropdown(
        options=possible_partitions,
        layout = widgets.Layout(width='200px')
        )

    def partition_change(_):
        if _dropdown_partition_on.value is None:
            _button_train.partitions = {"__dataset__": XRegressor()}
            return
        partitioned_model.partition_on = _dropdown_partition_on.value

        parts = {
            p: XRegressor() for p in df[_dropdown_partition_on.value].unique()}
        
        _button_train.partitions.update(parts)

    _dropdown_partition_on.observe(partition_change, names=['value'])

    # get all cols with cardinality of 1
    potential_id_cols = [None] + [
        col for col in df.columns if XScan._cardinality(df[col]) == 1]

    _selector_id_columns = widgets.SelectMultiple(
        options=potential_id_cols,
        style=style,
        layout = widgets.Layout(width='200px')
        )

    _title_partition_on = widgets.HTML(f"<h5>Partition on</h5>")

    _col_selectors = widgets.VBox([
        _title_target,
        _dropdown_target,
        _title_id,
        _selector_id_columns,
        _title_partition_on,
        _dropdown_partition_on
    ])
    
    vertical_divider = widgets.HTML(
            value='''<div style="border-left: 1px solid #cccccc;
                height: 100%; margin: 0 8px;"></div>''',
            layout=widgets.Layout(height='auto', margin='0 0 0 10px')
        )
    
    # PARAMETERS
    
    _title_parameters = widgets.HTML(f"<h5>Hyperparameters</h5>")
    
    _slider_max_depth = widgets.IntSlider(
        description='max_depth ', min=2, max=30, value=8, style=style)
    
    _slider_min_leaf_size = widgets.FloatSlider(
        description='min_leaf_size ', min=0.0001,
        max=0.2, value=0.025, step=0.0001, style=style,
        readout_format='.4f')
    
    _slider_min_info_gain = widgets.FloatSlider(
        description='min_info_gain ', min=0.0001, max=0.2,
        value=0.015, step=0.0001, style=style, readout_format='.4f')
    
    _slider_tail_sensitivity = widgets.FloatSlider(
        description='tail_sensitivity ', min=1, max=2,
        value=1, step=0.01, style=style, readout_format='.2f',
        disabled=True
    )
    
    _toggle_optimise_ts = widgets.Checkbox(description='opt', value=True)
    
    _slider_tail_sensitivity.layout.width = '300px'
    _toggle_optimise_ts.layout.width = '150px'
    _toggle_optimise_ts.layout.margin = '3px 0 0 -100px'
    
    _box_tail_sensitivity = widgets.HBox(
        [_slider_tail_sensitivity, _toggle_optimise_ts]
    )
    
    def on_optimise_ts(cb):
        if _toggle_optimise_ts.value:
            _slider_tail_sensitivity.disabled = True
        else:
            _slider_tail_sensitivity.disabled = False
            
    _toggle_optimise_ts.observe(on_optimise_ts, names=['value'])
    
    _title_settings = widgets.HTML(f"<h5>Settings</h5>")
    
    _slider_alpha = widgets.FloatSlider(
        description='alpha ', min=0.01, max=1, value=0.05,
        step=0.01, style=style, readout_format='.2f')
    
    _slider_opt_sample = widgets.FloatSlider(
        description='optimiser subsample ', min=0.05, max=1, value=1,
        step=0.01, style=style, readout_format='.2f')
    
    _slider_validation_size = widgets.FloatSlider(
        description='validation_size ', min=0.01, max=0.5, value=0.2,
        step=0.01, style=style, readout_format='.2f')
    
    _parameters = widgets.VBox([
        _title_parameters,
        _slider_max_depth,
        _slider_min_leaf_size,
        _slider_min_info_gain,
        _box_tail_sensitivity,
        divider,
        _title_settings, 
        _slider_alpha,
        _slider_validation_size
    ])
    
    partitioned_model = PartitionedRegressor(None)
    
    # OPTIMISATION
    
    xnet = XEvolutionaryNetwork(XRegressor())
    editor_output = widgets.Output()
    editor_output.layout.margin = '-10px 0 0 0'
    vp = VisualPipeline(xnet, editor_output=editor_output)
    vpipe = vp.show(hide_output=True)
    
    # Tighten
    mutations = widgets.IntSlider(
        min=2, max=200, value=50, description='mutations', style=style)
    
    generations = widgets.IntSlider(
        min=5, max=500, value=50, description='generations', style=style)
    
    max_generation_depth = widgets.IntSlider(
        min=5, max=100, value=10, description='max_generation_depth', style=style)
    
    max_severity = widgets.FloatSlider(
        min=0.01, max=0.8, value=0.3, step=0.01, description='max_severity',
        style=style)
    
    max_leaves = widgets.IntSlider(
        min=2, max=50, value=20, description='max_leaves', style=style)
    
    _box_evolve = widgets.VBox([
        mutations,
        generations,
        max_generation_depth,
        max_severity,
        max_leaves
    ])
    
    # Evolve
    iterations = widgets.IntSlider(
        min=10, max=1000, value=100, description='iterations', style=style)
    
    learning_rate = widgets.FloatSlider(
        min=0.001, max=0.5, value=0.1, step=0.001, description='learning_rate',
        style=style, readout_format='.3f')
    
    early_stopping = widgets.IntSlider(
        min=10, max=1000, value=100, description='early_stopping', style=style)
    
    _box_tighten = widgets.VBox([
        iterations,
        learning_rate,
        early_stopping
    ])
    
    def generate_evolve_layer():
        layer = Evolve()
        layer.mutations = mutations.value
        layer.generations = generations.value
        layer.max_generation_depth = max_generation_depth.value
        layer.max_severity = round(max_severity.value, 3)
        layer.max_leaves = max_leaves.value
        
        return layer
    
    def generate_tighten_layer():
        layer = Tighten()
        layer.iterations = iterations.value
        layer.learning_rate = round(learning_rate.value, 3)
        layer.early_stopping = early_stopping.value
        
        return layer
    
    # Create and link stack
    optimisation_stack = widgets.Stack(
        [_box_tighten, _box_evolve]
    )
    
    layer_selector = widgets.ToggleButtons(options=['Tighten', 'Evolve'])
    layer_selector.style.button_width = '100px'
    layer_selector.style.button_height = '25px'

    widgets.jslink(
        (layer_selector, 'index'), (optimisation_stack, 'selected_index'))
    
    layer_selector.layout = widgets.Layout(margin = '0 0 12px 0')
    
    def on_add(_):
        
        if layer_selector.index == 0:
            layer = generate_tighten_layer()
            selector_idx = 0
            
        else:
            layer = generate_evolve_layer()
            selector_idx = 1

        def focus_box(b):
            layer_selector.index = selector_idx
            params = xnet.future_layers[vp._selected_index].params
            if b.description == 'edit':
            
                if 'iterations' in params:
                    iterations.value = params['iterations']
                    learning_rate.value = params['learning_rate']
                    early_stopping.value = params['early_stopping']
                else:
                    mutations.value = params['mutations']
                    generations.value = params['generations']
                    max_generation_depth.value = params['max_generation_depth']
                    max_severity.value = params['max_severity']
                    max_leaves.value = params['max_leaves']
                
                add_layer_button.layout.display = 'none'
                layer_selector.disabled = True
            
            else:
                if 'iterations' in params:
                    params['iterations'] = iterations.value
                    params['learning_rate'] = round(learning_rate.value, 3)
                    params['early_stopping'] = early_stopping.value
                else:
                    params['mutations'] = mutations.value
                    params['generations'] = generations.value
                    params['max_generation_depth'] = max_generation_depth.value
                    params['max_severity'] = round(max_severity.value, 3)
                    params['max_leaves'] = max_leaves.value

                vp.set_stage_attributes(vp._selected_index, params)
                add_layer_button.layout.display = 'block'
                layer_selector.disabled = False

        vp.add_stage(layer, on_click=focus_box)
        vp.on_click(focus_box)(vp.box.children[vp._selected_index].children[1])
        #vp.editor_output.clear_output()
    
    add_layer_button = widgets.Button(description='Add', icon='plus')
    add_layer_button.on_click(on_add)
    add_layer_button.layout.width = '75px'
    add_layer_button.layout.margin = '0 0 0 15px'
    add_layer_button.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
    
    _box_layer_buttons = widgets.HBox([layer_selector, add_layer_button])
    
    _title_layers = widgets.HTML(f"<h5>Layers</h5>")
    _title_set_params = widgets.HTML(f"<h6>Parameters</h6>")
    
    opt_tab = widgets.VBox([
        _title_settings,
        _slider_opt_sample,
        _title_layers,
        _box_layer_buttons,
        _title_set_params,
        optimisation_stack
    ]
        #layout=widgets.Layout(align_items='center')
    )
    
    tabs = widgets.Tab(
        [_parameters,opt_tab]
    )
    
    tabs.set_title(0, 'Parameters')
    tabs.set_title(1, 'Optimisation')
    tabs.layout = widgets.Layout(
        margin='0 75px 0 20px', width='400px', height='400px')
    
    def partition_change(_):
        if _dropdown_partition_on.value is None:
            _button_train.partitions = {"__dataset__": XRegressor()}
            return
        
        partitioned_model.partition_on = _dropdown_partition_on.value
        
        parts = {str(p): XRegressor() for p in \
                 df[_dropdown_partition_on.value].unique()}
        
        _button_train.partitions.update(parts)

    _dropdown_partition_on.observe(partition_change, names=['value'])

    def close_button_click(_):
        screen.close()
    
    def on_train(b):
        try:
            header.loader.start()
            parts = b.partitions
            p_on = _dropdown_partition_on.value
            
            body.close()
            footer.close()
            output_screen.layout.display = 'flex'
            
            part_progress.set_bounds(min_val=0, max_val=len(b.partitions))
            part_progress.set_suffix(f'/{len(b.partitions)}')
            part_progress.set_value('Partitions', 0)
            part_progress.expand_items(items=['Partitions'])
            
            eval_screens = {}
            
            for i, (p, model) in enumerate(parts.items(), start=1):
                
                callback.init()
                
                kvt.update_data({
                    'status': 'initialising',
                    'partition': p,
                    'layers': f'{len(xnet.future_layers)}'
                })
                
                network = copy.deepcopy(xnet)
                network.model = model

                model.params.update_parameters(
                        max_depth=_slider_max_depth.value,
                        min_leaf_size=_slider_min_leaf_size.value,
                        min_info_gain=_slider_min_info_gain.value,
                        tail_sensitivity=_slider_tail_sensitivity.value,
                        ignore_nan=False
                    )

                drop_cols = [_dropdown_target.value]

                if p != '__dataset__':
                    part = df[df[p_on] == p]

                    if len(part) < 100:
                        continue
                    
                    drop_cols.append(_dropdown_partition_on.value)
                    X = part.drop(columns=drop_cols)
                    y = part[_dropdown_target.value]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=_slider_validation_size.value,
                        random_state=1)
                    
                else:
                    
                    if _dropdown_partition_on.value is not None:
                                drop_cols.append(_dropdown_partition_on.value)

                    X, y = df.drop(columns=drop_cols), df[_dropdown_target.value]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=_slider_validation_size.value,
                        random_state=1)
                
                id_cols = [
                    i for i in list(_selector_id_columns.value) if i is not None]
                
                kvt.update_data({
                    'status': 'fitting model',
                    'partition': p,
                    'layers': f'{len(xnet.future_layers)}'
                })

                start = time.time()
                model.fit(
                    X_train, y_train, id_columns=id_cols, alpha=_slider_alpha.value)
                
                model.metadata['optimiser'] = {}
                
                # if _toggle_optimise_ts.value:
                #     kvt.update_data({
                #         'status': 'optimising tail_sensitivity',
                #         'partition': p,
                #         'layers': f'{len(xnet.future_layers)}'
                #     })

                #     start = time.time()
                #     #model.optimise_tail_sensitivity(X_train, y_train)
                #     elapsed = round(time.time()-start, 4)

                #     model.metadata['optimiser'][
                #         'tail_sensitivity_optimise_time'] = elapsed
                
                if len(network.future_layers) > 0:
                    kvt.update_data({
                        'status': 'optimising model weights',
                        'partition': p,
                        'layers': f'{len(xnet.future_layers)}'
                    })
                    
                    if _slider_opt_sample.value < 1:

                        X_train = X_train.sample(
                            int(len(X_train) * _slider_opt_sample.value))
                        
                        y_train = y_train.loc[X_train.index]
                    
                    start = time.time()
                    if _dropdown_partition_on.value is not None:
                        network.fit(
                            X_train,
                            y_train)
                    else:
                        network.fit(X_train, y_train)
                    
                    output_screen.children = (output_screen.children[0],) + \
                        (callback.group.show(),)
                    
                    network.optimise(callback=callback)

                    elapsed = round(time.time()-start, 4)
                    model.metadata['optimiser']['optimise_time'] = elapsed
                    model.metadata['optimiser']['layers'] = len(
                        network.completed_layers)

                    with bar_plot_output:
                        metric_bars.add_bar(p, network.checkpoint_score)
                
                partitioned_model.add_partition(model, p)
                
                e = EvaluateRegressor(model, X_train, y_train)
                        
                eval_screens[p] = e.profile(X_test, y_test)

                eval_items = {
                    'train': evaluate_regression(e.y, e.y_pred),
                    'validation': evaluate_regression(
                        e.y_val, e.y_val_pred)
                }
                model.metadata['evaluation'] = eval_items
                part_progress.set_value('Partitions', i)
                
            part_progress.collapse_items(items=['Partitions'])
            divider.close()
            header.title = {'title': 'Profile'}
            
            X, y = df.drop(
                columns=[_dropdown_target.value]), df[_dropdown_target.value]
            
            save = ModelPersist(partitioned_model, 'regression', X, y)
            
            partition_select = widgets.Dropdown(
                options = eval_screens.keys()
            )

            partition_select.layout = widgets.Layout(margin='10px 0 0 0')
            header.add_widget(partition_select)

            def show_evaluation():

                def _gen(partition=partition_select):
                    with result_out:
                        clear_output(wait=True)
                        display(eval_screens[partition])
                        display(save.save())
                
                w = widgets.interactive(_gen)
                w.children = (w.children[-1],)
                return w
            
            output_body.close()
            
            display(show_evaluation())
                
            header.loader.stop()

        except Exception as e:
            print(e)
            raise ValueError(e)
    
    # BODY
    body = widgets.HBox([_col_selectors, tabs, vpipe])
    
    # OUTPUT
    output_data = {}
    
    kvt = KeyValueTable(
        output_data,
        transpose=True,
        cell_alignment='center',
        header_alignment='center',
        table_width='350px'
    )
        
    callback = RegressionCallback(xnet)
    
    output_screen = widgets.VBox([
        kvt.html_widget
    ], layout=widgets.Layout(width='450px'))
    output_screen.layout.display = 'None'
    
    bar_plot_output = widgets.Output()
    metric_bars = IncrementalBarPlot(bar_plot_output)
    
    output_body = widgets.HBox([output_screen, bar_plot_output])
    
    # FOOTER
    _button_train = TrainButton(
        description='Train Model',
        partitions={"__dataset__": XRegressor()},
        icon='bolt',
        disabled=False
    )
    
    _button_train.layout = widgets.Layout(margin=' 10px 0 10px 20px')
    _button_train.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
    
    _button_train.on_click(on_train)

    _button_close = widgets.Button(description='Close')

    _button_close.layout = widgets.Layout(margin=' 10px 0 10px 10px')

    _button_close.on_click(close_button_click)

    footer = widgets.VBox(
        [divider, widgets.HBox([_button_train, _button_close])])

    screen = widgets.VBox([header.show(), body, footer, output_body, result_out])
    
    display(screen)
    
    return partitioned_model