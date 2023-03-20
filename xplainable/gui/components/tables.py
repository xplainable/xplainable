import ipywidgets as widgets
from IPython.display import display
from typing import Dict, List, Tuple


class KeyValueTable:
    def __init__(
        self,
        data: Dict[str, str],
        transpose: bool = False,
        padding: str = '1px',
        table_width: str = '50%',
        table_height: str = 'auto',
        header_alignment: str = 'left',
        cell_alignment: str = 'left',
        header_color: str = '#f2f2f2',
        border_color: str = '#dddddd',
        header_font_color: str = 'black',
        cell_font_color: str = 'black'
        ):

        self.data = data
        self.transpose = transpose
        self.padding = padding
        self.table_width = table_width
        self.table_height = table_height
        self.header_alignment = header_alignment
        self.cell_alignment = cell_alignment
        self.header_color = header_color
        self.border_color = border_color
        self.header_font_color = header_font_color
        self.cell_font_color = cell_font_color
        
        self.html_widget = widgets.HTML()
        self.update_html()

    def update_html(self) -> None:
        if self.transpose:
            table_header = f'<table style="border-collapse: collapse; width: {self.table_width}; height: {self.table_height}; font-family: Arial, sans-serif;">\n<tr>\n' + ''.join([f'<th style="border: 1px solid {self.border_color}; text-align: {self.header_alignment}; padding: {self.padding}; background-color: {self.header_color}; font-weight: bold; color: {self.header_font_color};">{key}</th>' for key in self.data.keys()]) + "\n</tr>"
            table_rows = "\n<tr>\n" + ''.join([f'<td style="border: 1px solid {self.border_color}; text-align: {self.cell_alignment}; padding: {self.padding}; color: {self.cell_font_color};">{value}</td>' for value in self.data.values()]) + "\n</tr>"
        else:
            table_header = f'<table style="border-collapse: collapse; width: {self.table_width}; height: {self.table_height}; font-family: Arial, sans-serif;">\n<tr>\n' + ''.join([f'<th style="border: 1px solid {self.border_color}; text-align: {self.header_alignment}; padding: {self.padding}; background-color: {self.header_color}; font-weight: bold; color: {self.header_font_color};">{key}</th><td style="border: 1px solid {self.border_color}; text-align: {self.cell_alignment}; padding: {self.padding}; color: {self.cell_font_color};">{value}</td></tr>\n' for key, value in self.data.items()])

        table_footer = """
        </table>
        """

        html_table = table_header + table_rows + table_footer if self.transpose else table_header + table_footer
        self.html_widget.value = html_table

    def display(self) -> None:
        display(self.html_widget)

    def update_data(self, data: Dict[str, str]) -> None:
        self.data = data
        self.update_html()
