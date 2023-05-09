import ipywidgets as widgets
from IPython.display import display
import drawsvg as draw
import xplainable as xp
from ...utils.xwidgets import offline_button, docs_button


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
        self._logo = widgets.HTML(self._build_logo(logo_size))
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

    def _build_logo(self, size):
        logo = draw.Drawing(200, 200)
        circles = [
            (100, 100, 14, "#E14067"),
            (124, 124, 11, "#0080EA"),
            (146, 146, 9.6, "#E14067"),
            (165, 165, 7, "#0080EA"),
            (184, 184, 4, "#E14067"),
            (76, 76, 11, "#0080EA"),
            (54, 54, 9.6, "#E14067"),
            (35, 35, 7, "#0080EA"),
            (16, 16, 4, "#E14067"),
            (124, 76, 11, "#0080EA"),
            (146, 54, 9.6, "#E14067"),
            (165, 35, 7, "#0080EA"),
            (184, 16, 4, "#E14067"),
            (76, 124, 11, "#0080EA"),
            (54, 146, 9.6, "#E14067"),
            (35, 165, 7, "#0080EA"),
            (16, 184, 4, "#E14067"),
            (100, 138, 9, "#E14067"),
            (138, 100, 9, "#E14067"),
            (100, 62, 9, "#E14067"),
            (62, 100, 9, "#E14067"),
            (162, 126, 7, "#E14067"),
            (178, 146, 5, "#E14067"),
            (188, 166, 4, "#E14067"),
            (38, 74, 7, "#E14067"),
            (22, 54, 5, "#E14067"),
            (12, 34, 4, "#E14067"),
            (162, 74, 7, "#E14067"),
            (178, 54, 5, "#E14067"),
            (188, 34, 4, "#E14067"),
            (38, 126, 7, "#E14067"),
            (22, 146, 5, "#E14067"),
            (12, 166, 4, "#E14067"),
            (74, 38, 7, "#E14067"),
            (54, 22, 5, "#E14067"),
            (34, 12, 4, "#E14067"),
            (126, 162, 7, "#E14067"),
            (146, 178, 5, "#E14067"),
            (166, 188, 4, "#E14067"),
            (126, 38, 7, "#E14067"),
            (146, 22, 5, "#E14067"),
            (166, 12, 4, "#E14067"),
            (74, 162, 7, "#E14067"),
            (54, 178, 5, "#E14067"),
            (34, 188, 4, "#E14067"),
        ]

        for cx, cy, r, fill in circles:
            logo.append(draw.Circle(cx, cy, r, fill=fill))

        logo.set_render_size(size)
        
        return logo.as_html()
    
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

