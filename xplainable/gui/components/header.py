import ipywidgets as widgets
from IPython.display import display
import xplainable as xp
from ...utils.xwidgets import offline_button, docs_button
from ...utils.svgs import generate_xplainable_logo
from ...utils.loaders import LoadingPulse


class Header:
    
    def __init__(self, title='Title', logo_size=45, font_size=20, avatar=True):
        
        self.title_name = title
        self.font_size = font_size
        
        self._vwidgets = widgets.VBox([])
        self._hwidgets_left = widgets.HBox([], layout=widgets.Layout(
            display='flex', flex='1')
        )
        self.user = widgets.HBox([
            docs_button,
            xp.client.avatar if xp.client else offline_button
            ])
        self.user.layout = widgets.Layout(
            margin='auto', margin_right='15px', align_items = 'center'
        )
        self._hwidgets = widgets.HBox([self._hwidgets_left])
        if avatar:
            self._hwidgets.children += (self.user,)

        self._hwidgets.layout = widgets.Layout(
            display='flex', width='100%'
        )

        self.divider = widgets.HTML(
            f'<hr class="solid">',
            layout=widgets.Layout(margin='-15px 0 0 0')
        )
        
        self._title = self._generate_title(title, font_size)

        self.loader = LoadingPulse(
            size=logo_size, svg=generate_xplainable_logo(logo_size))
        self.loader.init()
        
        self._logo = self.loader.out
        
        self._header = None
        self._build_header()
    
    @property
    def title(self):
        return display(self._title)
    
    @title.setter
    def title(self, params):
        if 'title' in params:
            self.title_name = params.pop('title')
        
        if 'size' in params:
            self.font_size = params.pop('size')
        
        self._title.value = f'''
        <span style="font-family:Arial; font-size:{self.font_size}px;
        margin-left:10px;">{self.title_name}</span>'''

        if len(params) > 0:
            self._title.layout = widgets.Layout(**params)
        
    def _generate_title(self, title, font_size):
        title_widget = widgets.HTML(
            f'''<span style="font-family:Arial; font-size:{font_size}px;
            margin-left:10px;">{title}</span>''',

            layout=widgets.Layout(margin='11px 0 0 10px')
        )
        
        return title_widget
    
    def _build_header(self):
        
        if len(self._hwidgets_left.children) == 0:
            self._hwidgets_left.children += (
                self._logo,
                self._title
                )
    
        self._header = widgets.VBox(
            [self._hwidgets, self._vwidgets, self.divider]
        )
        
    
    def add_widget(self, widget, horizontal=True):
        if horizontal:
            self._hwidgets_left.children += (widget,)
        else:
            self._vwidgets.children += (widget,)
        
        return self
    
    def show(self):
        return self._header

