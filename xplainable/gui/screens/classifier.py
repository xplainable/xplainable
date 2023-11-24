""" Copyright Xplainable Pty Ltd, 2023"""

from IPython.display import display, clear_output
import ipywidgets as widgets
from ...core.models import XClassifier, PartitionedClassifier, ConstructorParams
from ...core.optimisation.bayesian import XParamOptimiser
from ...utils import TrainButton
from ...utils.activation import flex_activation
from ...quality import XScan
from ..components import Header
from .evaluate import EvaluateClassifier
from .save import ModelPersist
from ...metrics.metrics import evaluate_classification
from ...callbacks import OptCallback
from ..components import BarGroup
from ...utils.handlers import check_df

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
from ..._dependencies import _check_critical_versions

import os
os.environ['KMP_WARNINGS'] = '0'


def classifier(df):
    """ Trains an xplainable classifier via a simple GUI.

    Args:
        df (pandas.DataFrame): Training data including target

    Raises:
        RuntimeError: When model fails to fit

    Returns:
        xplainale.models.XClassifier: The trained model
    """

    # Check critical dependencies
    _check_critical_versions()

    # Assert dataframe is valid
    check_df(df)

    # This allows widgets to show full label text
    style = {'description_width': 'initial'}

    # Instantiate widget output
    outt = widgets.Output()
    #outt.layout = widgets.Layout(min_height='400px', display='none')

    # Instantiate the model
    partitioned_model = PartitionedClassifier(None)
    partitions = {"__dataset__": XClassifier()}

    divider = widgets.HTML(f'<hr class="solid">')

    # COLUMN 1
    col1a = widgets.HTML(
        f"<h5>Target</h5>", layout=widgets.Layout(height='auto'))

    id_title = widgets.HTML(f"<h5>ID Column (0 selected)</h5>")

    possible_targets = [None] + [i for i in df.columns if df[i].nunique() < 20]
    
    target = widgets.Dropdown(
        options=possible_targets,
        layout = widgets.Layout(width='200px')
        )

    possible_partitions = [None]+[i for i in df.columns if df[i].nunique() < 20]

    partition_on = widgets.Dropdown(
        options=possible_partitions,
        layout = widgets.Layout(width='200px')
        )

    def partition_change(_):
        if partition_on.value is None:
            train_button.partitions = {"__dataset__": XClassifier()}
            return
        partitioned_model.partition_on = partition_on.value
        parts = {p: XClassifier() for p in df[partition_on.value].unique()}
        train_button.partitions.update(parts)

    partition_on.observe(partition_change, names=['value'])

    # get all cols with cardinality of 1
    potential_id_cols = [None] + [
        col for col in df.columns if XScan._cardinality(df[col]) == 1]

    id_columns = widgets.SelectMultiple(
        options=potential_id_cols,
        style=style,
        layout = widgets.Layout(width='200px')
        )

    partition_header = widgets.HTML(f"<h5>Partition on</h5>")

    colA = widgets.VBox([
        col1a,
        target,
        id_title,
        id_columns,
        partition_header,
        partition_on])

    colA.layout = widgets.Layout(margin='0 0 0 15px')

    # COLUMN 2
    # Options to toggle param/opt view
    options = {
        True: 'flex',
        False: 'none'
    }

    # Change view on optimisation button change
    def optimise_changed(_):
        vala = optimise.value
        valb = not vala

        opt_params.layout.display = options[vala]
        std_params.layout.display = options[valb]
        optimise_metric.layout.display = options[vala]

    def target_changed(_):

        if target.value is None:
            train_button.disabled = True
        else:
            train_button.disabled = False

    optimise = widgets.Dropdown(
        value=False, options=[True, False],
        description='optimise:', style=style,
        layout = widgets.Layout(max_width='200px'))

    optimise_metrics = [
        'positive-f1',
        'negative-f1',
        'weighted-f1',
        'macro-f1',
        'positive-recall',
        'negative-recall',
        'weighted-recall',
        'macro-recall',
        'positive-precision',
        'negative-precision',
        'weighted-precision',
        'macro-precision',
        'accuracy',
        'log-loss',
        'brier-loss',
        'roc-auc'
    ]

    optimise_metric = widgets.Dropdown(
        value='roc-auc',
        options=optimise_metrics,
        description='metric:',
        style=style,
        layout = widgets.Layout(max_width='200px', margin='0 0 0 10px'))

    # Hide on instantiation
    optimise_metric.layout.display = 'none'

    optimise_display = widgets.HBox([
        optimise,
        optimise_metric
    ])

    optimise.observe(optimise_changed, names=['value'])
    target.observe(target_changed, names=['value'])

    # Param pickers
    max_depth = widgets.IntSlider(
        value=6, min=1, max=30, step=1, description='max_depth:',
        style=style)

    min_leaf_size = widgets.FloatSlider(
        value=0.025, min=0.0001, max=0.2, step=0.0001, readout_format='.4f',
        description='min_leaf_size:', style=style)

    min_info_gain = widgets.FloatSlider(
        value=0.015, min=0.0001, max=0.2, step=0.0001, readout_format='.4f',
        description='min_info_gain:', style=style)

    weight = widgets.FloatSlider(
        description='weight',
        style=style,
        value=0.0, min=0., max=3.0, step=0.01)
        
    power_degree = widgets.IntSlider(
        description='power_degree',
        style=style,
        value=1, min=1, max=5, step=2)
        
    sigmoid_exponent = widgets.FloatSlider(
        description='sigmoid_exponent',
        style=style,
        value=1.0, min=0.0, max=1.0, step=0.1)
    
    # Optimise param pickers
    n_trials = widgets.IntSlider(
        value=100, min=5, max=1000, step=5, description='n_trials:',
        style=style)

    early_stopping = widgets.IntSlider(
        value=40, min=5, max=1000, step=5, description='early_stopping:',
        style=style)

    # SEARCH SPACE – MAX_DEPTH
    max_depth_space = widgets.IntRangeSlider(
        value=[4, 10],
        min=2,
        max=30,
        step=1,
        description="max_depth:",
        style={'description_width': 'initial'},
        layout = widgets.Layout(min_width='350px')
    )

    max_depth_step = widgets.Dropdown(
        options=[1, 2, 5],
        value=2,
        layout = widgets.Layout(max_width='75px')
    )

    max_depth_space_display = widgets.HBox([max_depth_space, max_depth_step])

    # SEARCH SPACE – MIN_LEAF_SIZE
    min_leaf_size_space = widgets.FloatRangeSlider(
        value=[0.005, 0.05],
        min=0.0001,
        max=0.12,
        step=0.0005,
        description="min_leaf_size:",
        style={'description_width': 'initial'},
        readout_format='.4f',
        layout = widgets.Layout(min_width='350px')
    )

    min_leaf_size_step = widgets.Dropdown(
        options=[0.0005, 0.005, 0.01, 0.02],
        value=0.005,
        layout = widgets.Layout(max_width='75px')
    )

    min_leaf_size_display = widgets.HBox(
        [min_leaf_size_space, min_leaf_size_step])

    # SEARCH SPACE – MIN_LEAF_SIZE
    min_info_gain_space = widgets.FloatRangeSlider(
        value=[0.005, 0.05],
        min=0.0001,
        max=0.12,
        step=0.0005,
        description="min_info_gain:",
        style={'description_width': 'initial'},
        readout_format='.4f',
        layout = widgets.Layout(min_width='350px')
    )

    min_info_gain_step = widgets.Dropdown(
        options=[0.0005, 0.005, 0.01, 0.02],
        value=0.0005,
        layout = widgets.Layout(max_width='75px')
    )

    min_info_gain_display = widgets.HBox(
        [min_info_gain_space, min_info_gain_step])

    # SEARCH SPACE – WEIGHT
    weight_space = widgets.FloatRangeSlider(
        value=[0, 1.2],
        min=0,
        max=3,
        step=0.1,
        description="weight:",
        style={'description_width': 'initial'},
        readout_format='.2f',
        layout = widgets.Layout(min_width='350px')
    )

    weight_step = widgets.Dropdown(
        options=[0.05, 0.1, 0.25, 0.5],
        value=0.05,
        layout = widgets.Layout(max_width='75px')
    )

    weight_display = widgets.HBox(
        [weight_space, weight_step])

    # SEARCH SPACE - POWER_DEGREE
    power_degree_space = widgets.IntRangeSlider(
        value=[1, 3],
        min=1,
        max=5,
        step=2,
        description="power_degree:",
        style={'description_width': 'initial'},
        layout = widgets.Layout(min_width='350px')
    )

    power_degree_step = widgets.Dropdown(
        options=[2],
        value=2,
        layout = widgets.Layout(max_width='75px')
    )

    power_degree_display = widgets.HBox([power_degree_space, power_degree_step])

    # SEARCH SPACE – SIGMOID_EXPONENT
    sigmoid_exponent_space = widgets.FloatRangeSlider(
        value=[0.5, 1],
        min=0,
        max=1,
        step=0.1,
        description="sigmoid_exponent:",
        style={'description_width': 'initial'},
        readout_format='.2f',
        layout = widgets.Layout(min_width='350px')
    )

    sigmoid_exponent_step = widgets.Dropdown(
        options=[0.1, 0.25, 0.5],
        value=0.1,
        layout = widgets.Layout(max_width='75px')
    )

    sigmoid_exponent_display = widgets.HBox(
        [sigmoid_exponent_space, sigmoid_exponent_step])

    def plot_activation():
    
        @widgets.interactive
        def _activation(weight=weight, power_degree=power_degree,\
            sigmoid_exponent=sigmoid_exponent):

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

    activation_display = widgets.Output()
    with activation_display:
        display(plot_activation())

    hyperparameter_box = widgets.VBox([
        widgets.HTML(f"<h5>Hyperparameters</h5>"),
        max_depth,
        min_leaf_size,
        min_info_gain,
    ])

    activation_box = widgets.VBox([
        widgets.HTML(f"<h5>Activation</h5>"),
        weight,
        power_degree,
        sigmoid_exponent,
        activation_display
    ])

    parameter_box = widgets.HBox([hyperparameter_box, activation_box])

    std_params = widgets.VBox([
        parameter_box,
    ])

    opt_params = widgets.VBox([
        widgets.HTML(f"<h5>Trials</h5>"),
        n_trials,
        early_stopping,
        widgets.HTML(f"<h5>Search Space</h5>"),
        max_depth_space_display,
        min_leaf_size_display,
        min_info_gain_display,
        weight_display,
        power_degree_display,
        sigmoid_exponent_display

    ])

     # Set initial optimise widgets to no display
    opt_params.layout.display = 'none'

    colBParams = widgets.VBox([
        optimise_display,
        std_params,
        opt_params
        ])

    alpha_header = widgets.HTML(f"<h5>alpha</h5>")
    alpha = widgets.FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01)

    validation_size_header = widgets.HTML(f"<h5>Validation Size</h5>")
    validation_size = widgets.FloatSlider(
        value=0.2, min=0.05, max=0.5, step=0.01)

    colBSettings = widgets.VBox([
        alpha_header,
        alpha,
        validation_size_header,
        validation_size
        ])

    colB = widgets.Tab([colBParams, colBSettings])
    colB.set_title(0, 'Parameters')
    colB.set_title(1, 'Settings')
    colB.layout = widgets.Layout(margin='0 0 0 15px', min_width='400px')
    
    body = widgets.HBox([colA, colB])

    # HEADER
    header = Header(title='Train Classifier', logo_size=40, font_size=18)
    header.title = {'margin': '10px 15px 0 10px'}
    
    part_progress = BarGroup(items=['Partitions'])
    part_progress.collapse_items(items=['Partitions'])
    part_progress.window_layout.margin = '10px 0 0 0'

    header.add_widget(part_progress.show())

    # FOOTER
    train_button = TrainButton(
        description='Train Model', partitions=partitions, icon='bolt',
        disabled=True)
    
    train_button.layout = widgets.Layout(margin=' 10px 0 10px 20px')

    close_button = widgets.Button(description='Close')

    close_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')

    footer = widgets.VBox(
        [divider, widgets.HBox([train_button, close_button])])

    # SCREEN – Build final screen
    screen = widgets.VBox([header.show(), body, footer, outt])
    
    # Close and clear
    def close_button_click(_):
        screen.close()
        outt.close()
        clear_output()

    def id_cols_changed(_):
        id_vals = [i for i in list(id_columns.value) if i is not None]
        id_title.value = f"<h5>ID Column ({len(id_vals)} selected)</h5>"

    def generate_callback(
            _max_depth_space, _min_leaf_size_space, _min_info_gain_space,
            _weight_space, _power_degree_space, _sigmoid_exponent_space):

        opt_bars = BarGroup(
            items=['iteration', 'fold', 'best'],
            heading='Hyperparameter Optimiser'
            )

        opt_bars.bar_layout.height='20px'
        opt_bars.bar_layout.margin='6px 0 0 0'
        
        opt_bars.set_bounds(['iteration'], 0, n_trials.value)
        opt_bars.set_suffix(['iteration'], f'/{n_trials.value}')
        opt_bars.set_bar_color(color='#0080ea')

        # Params
        param_bars = BarGroup(
            items=[
            'max_depth',
            'min_leaf_size',
            'min_info_gain',
            'weight',
            'power_degree',
            'sigmoid_exponent'
            ],
            heading='Hyperparameters'
            )

        param_bars.bar_layout.height='20px'
        param_bars.bar_layout.margin='6px 0 0 0'
        param_bars.label_layout.width='110px'
        
        opt_bars.set_suffix(['fold'], '/5')
        opt_bars.set_prefix(['best'], f'{optimise_metric.value}: ')

        opt_bars.set_bounds(['fold'], 0, 5)
        opt_bars.set_bar_color(items=['fold'], color='#0080ea')
        opt_bars.set_bar_color(items=['best'], color='#12b980')

        param_bars.set_bounds(
            ['max_depth'], _max_depth_space[0], _max_depth_space[1])
        
        param_bars.set_bounds(
            ['min_leaf_size'], _min_leaf_size_space[0], _min_leaf_size_space[1])
        
        param_bars.set_bounds(
            ['min_info_gain'], _min_info_gain_space[0], _min_info_gain_space[1])

        param_bars.set_bounds(['weight'], _weight_space[0], _weight_space[1])

        param_bars.set_bounds(
            ['power_degree'], _power_degree_space[0], _power_degree_space[1])
        
        param_bars.set_bounds(
            ['sigmoid_exponent'], _sigmoid_exponent_space[0],
            _sigmoid_exponent_space[1])

        param_bars.set_bar_color(color='#e14067')

        opt_bars_display = opt_bars.show()
        param_bars_display = param_bars.show()
        opt_bars_display.layout = widgets.Layout(min_width='450px')
        param_bars_display.layout = widgets.Layout(margin='0 0 0 50px')

        callback = OptCallback(opt_bars, param_bars)

        return callback, opt_bars_display, param_bars_display

    # Train model on click
    def train_button_clicked(b):
        
        # Drop partitions with only one class
        for p in list(b.partitions.keys()):
            if p != '__dataset__':
                if df[df[partition_on.value] == p][target.value].nunique() < 2:
                    b.partitions.pop(p)
                    
        header.loader.start()

        part_progress.set_bounds(min_val=0, max_val=len(b.partitions))
        part_progress.set_suffix(f'/{len(b.partitions)}')
        part_progress.set_value('Partitions', 0)
        part_progress.expand_items(items=['Partitions'])

        with outt:
            
            parts = b.partitions
            body.close()
            footer.close()
            #outt.layout.display = 'flex'

            eval_screens = {}

            if optimise.value:
                _max_depth_space = list(max_depth_space.value) + \
                    [max_depth_step.value]
                _max_depth_space[1] += 0.0001
                
                _min_leaf_size_space = list(min_leaf_size_space.value) + \
                    [min_leaf_size_step.value]
                _min_leaf_size_space[1] += 0.0001

                _min_info_gain_space = list(min_info_gain_space.value) + \
                    [min_info_gain_step.value]
                _min_info_gain_space[1] += 0.0001

                _weight_space = list(weight_space.value) + \
                    [weight_step.value]
                _weight_space[1] += 0.0001

                _power_degree_space = list(power_degree_space.value) + \
                    [power_degree_step.value]
                _power_degree_space[1] += 0.0001

                _sigmoid_exponent_space = list(sigmoid_exponent_space.value) + \
                    [sigmoid_exponent_step.value]
                _sigmoid_exponent_space[1] += 0.0001

                callback, opt_bars_display, param_bars_display = generate_callback(
                _max_depth_space, _min_leaf_size_space, _min_info_gain_space,
                _weight_space, _power_degree_space, _sigmoid_exponent_space
                )

                desc_partition = widgets.HTML(f'<strong>Partition: </strong>-')

                desc_optimise_on = widgets.HTML(
                    f'<strong>Optimise on: </strong>{optimise_metric.value}')
                
                desc_early_stopping = widgets.HTML(
                    f'<strong>Early Stopping: </strong>{early_stopping.value}')

                opt_details = widgets.VBox([
                    opt_bars_display,
                    divider,
                    desc_partition,
                    desc_optimise_on,
                    desc_early_stopping
                ])

                opt_display = widgets.HBox([opt_details, param_bars_display])
                display(opt_display)

            for i, (p, model) in enumerate(parts.items()):

                model.params.update_parameters(
                    max_depth=max_depth.value,
                    min_leaf_size=min_leaf_size.value,
                    min_info_gain=min_info_gain.value,
                    weight=weight.value,
                    power_degree=power_degree.value,
                    sigmoid_exponent=sigmoid_exponent.value,
                    ignore_nan=False
                )

                try:
                    if optimise.value:
                        desc_partition.value = f'<strong>Partition: </strong>{p}'
                        callback.reset()
                    
                    drop_cols = [target.value]

                    if p != '__dataset__':
                        part = df[df[partition_on.value] == p]
        
                        if len(part) < 100:
                            continue
                        drop_cols.append(partition_on.value)
                        X, y = part.drop(columns=drop_cols), part[target.value]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=validation_size.value,
                            random_state=1)
                        
                    else:
                    
                        if partition_on.value is not None:
                            drop_cols.append(partition_on.value)

                        X, y = df.drop(columns=drop_cols), df[target.value]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=validation_size.value,
                            random_state=1)

                    id_cols = [
                        i for i in list(id_columns.value) if i is not None]
                    
                    if optimise.value:
      
                        opt = XParamOptimiser(
                            metric = optimise_metric.value,
                            early_stopping=early_stopping.value,
                            n_trials=n_trials.value,
                            n_folds=5,
                            alpha=alpha.value,
                            max_depth_space=_max_depth_space,
                            min_leaf_size_space=_min_leaf_size_space,
                            min_info_gain_space=_min_info_gain_space,
                            weight_space=_weight_space,
                            power_degree_space=_power_degree_space,
                            sigmoid_exponent_space=_sigmoid_exponent_space,
                            verbose=False
                        )
                        
                        # This returns an optimised model
                        params = opt.optimise(X_train, y_train, verbose=False,
                                     callback=callback)
                        
                        model.set_params(ConstructorParams(**params))
                        model.metadata["optimised"] = True
                        model.metadata["optimiser"] = opt.metadata

                    model.fit(X_train, y_train, id_columns=id_cols)

                    partitioned_model.add_partition(model, p)
                    
                    if len(model.target_map) > 0:
                        if y_train.dtype == 'object':
                            y_train = y_train.map(model.target_map)
                        if y_test.dtype == 'object':
                            y_test = y_test.map(model.target_map)

                    e = EvaluateClassifier(model, X_train, y_train)
                    
                    eval_screens[p] = e.profile(X_test, y_test)

                    eval_items = {
                        'train': evaluate_classification(e.y, e.y_prob),
                        'validation': evaluate_classification(
                            e.y_val, e.y_val_prob)
                    }

                    model.metadata['evaluation'] = eval_items
                    part_progress.set_value('Partitions', i+1)
                    
                except Exception as e:
                    # screen.close()
                    # clear_output()
                    raise RuntimeError(e)
            
            if optimise.value:
                callback.finalise()
                opt_display.close()
            
            part_progress.collapse_items(items=['Partitions'])
            header.title = {'title': 'Profile'}
            divider.close()

            X, y = df.drop(columns=[target.value]), df[target.value]
            save = ModelPersist(
                partitioned_model, 'binary_classification', X, y)

            partition_select = widgets.Dropdown(
                options = eval_screens.keys()
            )

            partition_select.layout = widgets.Layout(margin='10px 0 0 0')

            header.add_widget(partition_select)

            def show_evaluation():
                
                def _gen(partition=partition_select):
                    display(eval_screens[partition])
                w = widgets.interactive(_gen)
                w.children = (w.children[-1],)
                return w

            display(show_evaluation())
            display(save.save())
            header.loader.stop()
        
    # Listen for clicks
    train_button.on_click(train_button_clicked)
    train_button.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }
            
    close_button.on_click(close_button_click)

    # Listen for changes
    id_columns.observe(id_cols_changed, names=['value'])

    # Display screen
    display(screen)

    # Need to return empty model first
    return partitioned_model
