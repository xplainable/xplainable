import ipywidgets as widgets
from IPython.display import display

class CardWidget:
    def __init__(
        self,
        title,
        value,
        title_size='12px',
        value_size='24px',
        card_size='150px',
        rounded_edges=True,
        background_color='#ffffff',
        title_color='#4e5c6c',
        value_color='#1b1e1d',
        title_align='center',
        value_align='center'
        ):

        self.title = title
        self.value = value
        self.title_size = title_size
        self.value_size = value_size
        self.card_size = card_size
        self.rounded_edges = rounded_edges
        self.background_color = background_color
        self.title_color = title_color
        self.value_color = value_color
        self.title_align = title_align
        self.value_align = value_align

        self.card = self._create_card()

    def _create_card(self):
        card_style = f"""
            background-color: {self.background_color};
            padding: 16px;
            border-radius: {'12px' if self.rounded_edges else '0px'};
            width: {self.card_size};
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        """

        title_style = f"""
            font-size: {self.title_size};
            color: {self.title_color};
            margin: 0;
            margin-bottom: 8px;
            text-align: {self.title_align};
        """

        value_style = f"""
            font-size: {self.value_size};
            color: {self.value_color};
            margin: 0;
            text-align: {self.value_align};
        """

        card = widgets.HTML(
            value=f"""
                <div style='{card_style}'>
                    <h3 style='{title_style}'>{self.title}</h3>
                    <p style='{value_style}'>{self.value}</p>
                </div>
            """
        )
        return card

    def set_value(self, new_value):
        self.value = new_value
        self.card.value = self._create_card().value

    def hide_card(self):
        self.card.layout.visibility = 'hidden'

    def show_card(self):
        self.card.layout.visibility = 'visible'

    def display(self):
        display(self.card)
