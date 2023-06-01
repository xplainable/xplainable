from ...utils.xwidgets import LayerButton
import ipywidgets as widgets
from IPython.display import display, clear_output

class VisualPipeline:
    
    def __init__(self, network, title='Pipeline', editor_output=None):
        self.network = network
        self.title = title
        
        self._widgets = []
        self.box = widgets.VBox([])
        self.init()
        self._selected_index = 0
        
        if editor_output is None:
            self.editor_output = widgets.Output()
        else:
            self.editor_output = editor_output
        
    def init(self):
        self.__arrow = widgets.HTML(
            '\u2193',
            layout=widgets.Layout(margin='-5px 0 -5px 0')
        )
        self.box_layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='center'
    )
        self.box.layout=self.box_layout
        
        self.accepted_params = [
            'iterations',
            'learning_rate',
            'early_stopping',
            'mutations',
            'generations',
            'max_generation_depth',
            'max_severity',
            'max_leaves'
        ]
    
    def on_click(self, edit_click):
        
        def callback(b):
            
            try:
                self.box.children[self._selected_index].children[1].style = {
                    "button_color": "#EEEEEE"
                }
            except:
                pass
            
            self._selected_index = b.idx
            
            self.box.children[b.idx].children[1].style = {
                "button_color": "#d6d4d4"
            }
            
            def toggle_edit(b):
                if b.description == 'edit':
                    b.description = 'done'
                else:
                    b.description = 'edit'
        
            def on_drop(_):
                self.drop_stage(b.idx)
                with self.editor_output:
                    clear_output()

            def on_up(_):
                self.move_up(b.idx)

            def on_down(_):
                self.move_down(b.idx)

            edit_button = widgets.Button(
                description = 'edit',
                layout=widgets.Layout(width='70px', height='27px')
            )

            delete_button = widgets.Button(
                description = 'drop',
                layout=widgets.Layout(width='70px', height='27px')
            )

            edit_button.style = {
                "button_color": '#0080ea',
                "text_color": 'white'
                }

            delete_button.style = {
                "button_color": '#e14067',
                "text_color": 'white'
                }

            up_button = widgets.Button(
                description='⬇',
                layout=widgets.Layout(width='40px', height='27px')
            )

            down_button = widgets.Button(
                description='⬆',
                layout=widgets.Layout(width='40px', height='27px')
            )

            edit_button.on_click(edit_click)
            edit_button.on_click(toggle_edit)
            delete_button.on_click(on_drop)
            up_button.on_click(on_up)
            down_button.on_click(on_down)

            editor = widgets.HBox(
                [edit_button, delete_button, down_button, up_button])

            with self.editor_output:
                clear_output(wait=True)
                display(editor)
                
        return callback
            
    def _append_child(self, child, idx):
        children = list(self.box.children)
        children.insert(idx, child)
        self.box.children = children
        
    def update_layer_text(self, idx):
        stage = self.box.children[idx].children[1].layer
        name = stage.__class__.__name__
        params = ", ".join([
            str(v) for i, v in stage.__dict__.items() if
            i in self.accepted_params])
        
        text = f'{name}({params})'
        
        self.box.children[idx].children[1].description = text
        
    def update_indices(self):
        for i, v in enumerate(list(self.box.children)):
            self.box.children[i].children[1].idx = i
        
    def add_stage(self, stage, idx=None, on_click=None):
        name = stage.__class__.__name__
        params = ", ".join([
            str(v) for i, v in stage.__dict__.items() if
            i in self.accepted_params])
        
        text = f'{name}({params})'
        
        idx = len(self.box.children) if idx is None else idx
        
        try:
            self.box.children[self._selected_index].children[1].style = {
                "button_color": "#EEEEEE"
            }
        except:
            pass
        
        button = LayerButton(
            description=text,
            idx=idx,
            layer=stage,
            layout=widgets.Layout(width='250px', height='27px')
            )
        
        if name not in ['XRegressor', 'XClassifier']:
            button.on_click(self.on_click(on_click))
            
        w = widgets.VBox([
            self.__arrow,
            button
        ], layout=self.box_layout)
        
        self._append_child(w, idx=idx)
        self.network.add_layer(stage, idx=idx)
        
        self.update_indices()
        self._selected_index = len(self.network.future_layers)-1
        
        self.box.children[self._selected_index].children[1].style = {
                "button_color": "#d6d4d4"
            }
        
    def set_stage_attributes(self, idx, attr: dict):
        for i, v in attr.items():
            setattr(self.network.future_layers[idx], i, v)
        
        self.update_layer_text(idx)
    
    def drop_stage(self, idx):
        children = list(self.box.children)
        del children[idx]
        del self.network.future_layers[idx]
        self.box.children = children
        self.update_indices()
        
    def get_pipeline(self):
                    
        return self.network.future_layers
    
    @staticmethod
    def shift_right(lst):
        if len(lst) == 0:
            return lst
        return [lst[-1]] + lst[:-1]

    @staticmethod
    def shift_left(lst):
        if len(lst) == 0:
            return lst
        return lst[1:] + [lst[0]]

    def shift_index_right(self, lst, index):
        if len(lst) == 0:
            return lst
        if index == len(lst) - 1:
            return self.shift_right(lst)
        lst[index], lst[index + 1] = lst[index + 1], lst[index]
        return lst
    
    def shift_index_left(self, lst, index):
        if len(lst) == 0:
            return lst
        if index == 0:
            return self.shift_left(lst)
        lst[index], lst[index - 1] = lst[index - 1], lst[index]
        return lst
        
    def move_down(self, idx):
               
        self.network.future_layers = self.shift_index_left(
            self.network.future_layers, idx)
        
        children = list(self.box.children)
        children = self.shift_index_left(children, idx)        
        self.box.children = children
        
        self.update_indices()
        
        if idx == 0:
            self._selected_index = len(children)-1
        else:
            self._selected_index = idx - 1
    
    def move_up(self, idx):
               
        self.network.future_layers = self.shift_index_right(
            self.network.future_layers, idx)
        
        children = list(self.box.children)
        children = self.shift_index_right(children, idx)        
        self.box.children = children
    
        self.update_indices()
        if idx == len(children) - 1:
            self._selected_index = 0
        else:
            self._selected_index = idx + 1
            
    def show(self, hide_output=False):
        
        divider = widgets.HTML(f'<hr class="solid">')
        
        model_name = self.network.model.__class__.__name__

        base_stage = LayerButton(
            description=model_name,
            idx=0,
            layer=None,
            layout=widgets.Layout(
                width='250px', height='27px', align_items='center'),
            disabled=True
            )
        
        scroll_layers = widgets.VBox([
            base_stage,
            self.box
            ],
            layout=widgets.Layout(align_items='center')
        )
        
        pipe = widgets.VBox([
            widgets.HTML(f"<h4>{self.title}</h4>" ),
            scroll_layers,
            divider,
            self.editor_output,
        ], layout=self.box_layout)
        
        scroll_layers.layout.max_height = '280px'
        scroll_layers.layout.overflow_y = 'auto'
        scroll_layers.layout.display='block'
        scroll_layers.layout.align_items='center'
        
        return pipe
