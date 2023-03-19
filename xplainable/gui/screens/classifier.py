from IPython.display import display, clear_output
import ipywidgets as widgets
from ...core.models import Classifier
from ...core.optimisation.bayesian import XParamOptimiser
from ...utils import TrainButton
from ...quality import XScan
import time

#import ray
#ray.init()


from ...callbacks import OptCallback, OptCallbackRay
from ..components import BarGroup

def classifier(df):
    """ Trains an xplainable classifier via a simple GUI.

    Args:
        df (pandas.DataFrame): Training data including target

    Raises:
        RuntimeError: When model fails to fit

    Returns:
        xplainale.models.XClassifier: The trained model
    """

    # This allows widgets to show full label text
    style = {'description_width': 'initial'}

    # Instantiate widget output
    outt = widgets.Output()

    # Instantiate the model
    model = Classifier()

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

    possible_partitions = [None]+[i for i in df.columns if df[i].nunique() < 11]

    partition_on = widgets.Dropdown(
        options=possible_partitions,
        layout = widgets.Layout(width='200px')
        )

    # get all cols with cardinality of 1
    potential_id_cols = [None] + [
        col for col in df.columns if XScan._cardinality(df[col]) == 1]

    id_columns = widgets.SelectMultiple(
        options=potential_id_cols,
        style=style,
        layout = widgets.Layout(width='200px')
        )

    colA = widgets.VBox([col1a, target, id_title, id_columns])
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
        'weighted-f1',
        'macro-f1',
        'accuracy',
        'recall',
        'precision'
    ]

    optimise_metric = widgets.Dropdown(
        value='weighted-f1',
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
        value=12, min=2, max=100, step=1, description='max_depth:',
        style=style)

    min_leaf_size = widgets.FloatSlider(
        value=0.015, min=0.001, max=0.2, step=0.001, readout_format='.3f',
        description='min_leaf_size:', style=style)

    min_info_gain = widgets.FloatSlider(
        value=0.015, min=0.001, max=0.2, step=0.001, readout_format='.3f',
        description='min_info_gain:', style=style)
    
    # Optimise param pickers
    n_trials = widgets.IntSlider(
        value=30, min=5, max=150, step=5, description='n_trials:',
        style=style)

    early_stopping = widgets.IntSlider(
        value=15, min=5, max=50, step=5, description='early_stopping:',
        style=style)

    # SEARCH SPACE – MAX_DEPTH
    max_depth_space = widgets.IntRangeSlider(
        value=[4, 22],
        min=2,
        max=100,
        step=1,
        description="max_depth:",
        style={'description_width': 'initial'},
        layout = widgets.Layout(min_width='350px')
    )

    max_depth_step = widgets.Dropdown(
        options=[1, 2, 5],
        layout = widgets.Layout(max_width='75px')
    )

    max_depth_space_display = widgets.HBox([max_depth_space, max_depth_step])

    # SEARCH SPACE – MIN_LEAF_SIZE
    min_leaf_size_space = widgets.FloatRangeSlider(
        value=[0.005, 0.08],
        min=0.005,
        max=0.2,
        step=0.005,
        description="min_leaf_size:",
        style={'description_width': 'initial'},
        readout_format='.3f',
        layout = widgets.Layout(min_width='350px')
    )

    min_leaf_size_step = widgets.Dropdown(
        options=[0.005, 0.01, 0.02],
        layout = widgets.Layout(max_width='75px')
    )

    min_leaf_size_display = widgets.HBox(
        [min_leaf_size_space, min_leaf_size_step])

    # SEARCH SPACE – MIN_LEAF_SIZE
    min_info_gain_space = widgets.FloatRangeSlider(
        value=[0.005, 0.08],
        min=0.005,
        max=0.2,
        step=0.005,
        description="min_info_gain:",
        style={'description_width': 'initial'},
        readout_format='.3f',
        layout = widgets.Layout(min_width='350px')
    )

    min_info_gain_step = widgets.Dropdown(
        options=[0.005, 0.01, 0.02],
        layout = widgets.Layout(max_width='75px')
    )

    min_info_gain_display = widgets.HBox(
        [min_info_gain_space, min_info_gain_step])

    std_params = widgets.VBox([
        widgets.HTML(f"<h5>Hyperparameters</h5>"),
        max_depth,
        min_leaf_size,
        min_info_gain
    ])

    opt_params = widgets.VBox([
        widgets.HTML(f"<h5>Trials</h5>"),
        n_trials,
        early_stopping,
        widgets.HTML(f"<h5>Search Space</h5>"),
        max_depth_space_display,
        min_leaf_size_display,
        min_info_gain_display
    ])

     # Set initial optimise widgets to no display
    opt_params.layout.display = 'none'

    colBParams = widgets.VBox([
        optimise_display,
        std_params,
        opt_params
        ])

    partition_header = widgets.HTML(f"<h5>Partition on</h5>")

    alpha_header = widgets.HTML(f"<h5>alpha</h5>")
    alpha = widgets.FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01)

    validation_size_header = widgets.HTML(f"<h5>Validation Size</h5>")
    validation_size = widgets.FloatSlider(
        value=0.2, min=0.05, max=0.5, step=0.01)

    colBSettings = widgets.VBox([
        partition_header,
        partition_on,
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

    # FOOTER
    train_button = TrainButton(
        description='Train Model', model=model, icon='bolt', disabled=True)

    close_button = widgets.Button(description='Close')

    train_button.layout = widgets.Layout(margin=' 10px 0 10px 20px')
    close_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')

    footer = widgets.VBox(
        [divider, widgets.HBox([train_button, close_button])])

    title_text = widgets.HTML('<h3>Train Classifier<h3>')
    title_text.layout = widgets.Layout(margin='0 0 0 20px')
    title = widgets.VBox([title_text, divider])

    # SCREEN – Build final screen
    screen = widgets.VBox([title, body, footer, outt])
    
    # Close screen
    def close():
        title_text.close()
        body.close()
        footer.close()

    # Close and clear
    def close_button_click(_):
        close()
        clear_output()

    def id_cols_changed(_):
        id_vals = [i for i in list(id_columns.value) if i is not None]
        id_title.value = f"<h5>ID Column ({len(id_vals)} selected)</h5>"

    def generate_callback(_max_depth_space, _min_leaf_size_space, _min_info_gain_space):
        opt_bars = BarGroup(
            items=['iteration'],
            heading='Hyperparameter Optimiser'
            )

        opt_bars.bar_layout.height='20px'
        opt_bars.bar_layout.margin='6px 0 0 0'

        
        opt_bars.set_bounds(['iteration'], 0, n_trials.value)
        opt_bars.set_suffix(['iteration'], f'/{n_trials.value}')
        opt_bars.set_bar_color(color='#0080ea')
        opt_bars.add_button(text='info', side='right')

        # Params
        param_bars = BarGroup(
            items=['fold', 'max_depth', 'min_leaf_size', 'min_info_gain', 'best'],
            heading='Hyperparameters'
            )

        param_bars.bar_layout.height='20px'
        param_bars.bar_layout.margin='6px 0 0 0'
        param_bars.label_layout.width='90px'
        
        param_bars.set_suffix(['fold'], '/5')
        param_bars.set_prefix(['best'], f'{optimise_metric.value}: ')

        param_bars.set_bounds(['fold'], 0, 5)
        param_bars.set_bounds(['max_depth'], _max_depth_space[0], _max_depth_space[1])
        param_bars.set_bounds(['min_leaf_size'], _min_leaf_size_space[0], _min_leaf_size_space[1])
        param_bars.set_bounds(['min_info_gain'], _min_info_gain_space[0], _min_info_gain_space[1])

        param_bars.set_bar_color(color='#e14067')
        param_bars.set_bar_color(items=['fold'], color='#0080ea')
        param_bars.set_bar_color(items=['best'], color='#12b980')

        opt_bars_display = opt_bars.show()
        param_bars_display = param_bars.show()
        opt_bars_display.layout = widgets.Layout(min_width='400px')
        param_bars_display.layout = widgets.Layout(margin='0 0 0 50px')

        callback = OptCallback(opt_bars, param_bars)

        return callback, opt_bars_display, param_bars_display

    # Train model on click
    def train_button_clicked(b):

        with outt:
        
            model = b.model
            model.max_depth = max_depth.value
            model.min_leaf_size = min_leaf_size.value
            model.min_info_gain = min_info_gain.value
            model.alpha = alpha.value

            try:
                body.close()
                footer.close()
                
                id_cols = [i for i in list(id_columns.value) if i is not None]
                X, y = df.drop(columns=[target.value]), df[target.value]

                #tabs = widgets.Tab(children=[widgets.Output()])
                #tabs.set_title(0, '__dataset__')
                #display(tabs)

                #ray_tasks = []
                #ray_callbacks = []
                #callbacks = []

                if optimise.value:

                    _max_depth_space = list(max_depth_space.value) + \
                        [max_depth_step.value]

                    _min_leaf_size_space = list(min_leaf_size_space.value) + \
                        [min_leaf_size_step.value]

                    _min_info_gain_space = list(min_info_gain_space.value) + \
                        [min_info_gain_step.value]
                            
                    opt = XParamOptimiser(
                        metric = optimise_metric.value,
                        early_stopping=early_stopping.value,
                        n_trials=n_trials.value,
                        n_folds=5,
                        max_depth_space=_max_depth_space,
                        min_leaf_size_space=_min_leaf_size_space,
                        min_info_gain_space=_min_info_gain_space,
                        verbose=False
                    )

                    callback, opt_bars_display, param_bars_display = generate_callback(
                        _max_depth_space, _min_leaf_size_space, _min_info_gain_space)

                    #callbacks.append(callback)

                    #ray_callback = OptCallbackRay()
                    #ray_callbacks.append(ray_callback)
                    display(widgets.HBox([opt_bars_display, param_bars_display]))
                
                    #ray_tasks.append(opt.ray_optimise.remote(X, y, verbose=False, callback=ray_callback))
                    params = opt.optimise(X, y, verbose=False, callback=callback)
                    model.set_params(**params)

                model.fit(X, y, id_columns=id_cols)
                
                # if partition_on.value is not None:
                    
                #     unq = list(df[partition_on.value].unique())

                #     tabs.children = tabs.children + tuple([widgets.Output() for i in range(len(unq))])
                #     for i, u in enumerate(unq):
                #         tabs.set_title(i+1, u)
                    
                #     for idx, i in enumerate(unq):
                #         part = df[df[partition_on.value] == i].copy()
                #         X, y = part.drop(columns=[target.value, partition_on.value]), part[target.value]

                #         if optimise.value:

                #             _max_depth_space = list(max_depth_space.value) + \
                #                 [max_depth_step.value]

                #             _min_leaf_size_space = list(min_leaf_size_space.value) + \
                #                 [min_leaf_size_step.value]

                #             _min_info_gain_space = list(min_info_gain_space.value) + \
                #                 [min_info_gain_step.value]
                                    
                #             opt = XParamOptimiser(
                #                 metric = optimise_metric.value,
                #                 early_stopping=early_stopping.value,
                #                 n_trials=n_trials.value,
                #                 n_folds=5,
                #                 max_depth_space=_max_depth_space,
                #                 min_leaf_size_space=_min_leaf_size_space,
                #                 min_info_gain_space=_min_info_gain_space,
                #                 verbose=False
                #             )

                #             callback, opt_bars_display, param_bars_display = generate_callback(
                #                 _max_depth_space, _min_leaf_size_space, _min_info_gain_space)
                #             callbacks.append(callback)

                #             #ray_callback = OptCallbackRay()
                #             #ray_callbacks.append(ray_callback)
                #             with tabs.children[idx+1]:
                #                 display(widgets.HBox([opt_bars_display, param_bars_display]))
                                
                #                 #ray_tasks.append(opt.ray_optimise.remote(X, y, verbose=False, callback=ray_callback))
                #                 params = opt.optimise(X, y, verbose=False, callback=callback)

                #                 sub_model = Classifier(**params)
                #                 sub_model.fit(X, y, id_columns=id_cols)
                    
                    # while any([ray.get(c._iteration.to_value.remote()) < n_trials.value for c in ray_callbacks]):
                    #     for cb, rcb in zip(callbacks, ray_callbacks):
                    #         cb.fold = ray.get(rcb._fold.to_value.remote())
                    #         cb.iteration = ray.get(rcb._iteration.to_value.remote())
                    #         cb.metric = ray.get(rcb._metric.to_value.remote())
                    #         md = ray.get(rcb._max_depth.to_value.remote())
                    #         mls = ray.get(rcb._min_leaf_size.to_value.remote())
                    #         mig = ray.get(rcb._min_info_gain.to_value.remote())
                    #         cb.update_params(md, mls, mig)
                    #         time.sleep(0.01)

                    # results = ray.get(ray_tasks)
                    # X, y = df.drop(columns=[target.value]), df[target.value]
                    # sub_model = Classifier(**results[idx+1])
                    # id_cols = [i for i in list(id_columns.value) if i is not None]

                    # sub_model.fit(X, y, id_columns=id_cols)
                    # print('trained')

                    # for idx, i in enumerate(unq):
                    #     part = df[df[partition_on.value] == i].copy()
                    #     X, y = part.drop(columns=[target.value, partition_on.value]), part[target.value]

                    #     sub_model = Classifier(**results[idx+1])
                    #     id_cols = [i for i in list(id_columns.value) if i is not None]

                    #     sub_model.fit(X, y, id_columns=id_cols)
                    #     print('trained')

            except Exception as e:
                close()
                clear_output()
                raise RuntimeError(e)

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
    return model