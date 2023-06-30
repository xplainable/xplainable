from IPython.display import display
import requests
from IPython.core.display import HTML
import base64
from ...utils.handlers import add_thousands_separator
import ipywidgets as widgets


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
        self.value = add_thousands_separator(value)
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
        self.value = add_thousands_separator(new_value)
        self.card.value = add_thousands_separator(self._create_card().value)

    def hide_card(self):
        self.card.layout.visibility = 'hidden'

    def show_card(self):
        self.card.layout.visibility = 'visible'

    def display(self):
        display(self.card)


def render_user_avatar(data):
    try:
        response = requests.get(data['image'])
        img_data = base64.b64encode(response.content).decode('utf-8')
    except:
        img_data = ''
        initials = f"{data['given_name'][0]}{data['family_name'][0]}"
        
    name = f"{data['given_name']} {data['family_name']}"

    html_card = f'''
        <style>
            .rounded-circle {{
                border-radius: 50% !important;
                width: 35px;
                height: 35px;
                object-fit: cover;
            }}
            .avatar-container {{
                display: flex;
                margin-right: 15px;
            }}
            .avatar-image {{
                display: flex;
                align-items: center;
            }}
            .username-container {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-left: 10px;
            }}
            .username {{
                font-size: 16px;
                margin-top: 3px;
                margin-bottom: -10px;
            }}
            .position {{
                font-size: 12px;
                color: grey;
                margin-left: 2px;
            }}
            .default-avatar {{
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #ccc;
                width: 35px;
                height: 35px;
                border-radius: 50%;
                font-size: 14px;
                font-weight: bold;
                color: #fff;
            }}
        </style>

        <div class="avatar-container">
            <div class="avatar-image">
                {f'<img src="data:image/jpeg;base64,{img_data}" class="rounded-circle">' if img_data else f'<div class="default-avatar">{initials}</div>'}
            </div>
            <div class="username-container">
                <span class="username">{name}</span>
                <span class="position">{data['position']}</span>
            </div>
        </div>
    '''
    
    avatar = widgets.HTML(html_card)
    return avatar