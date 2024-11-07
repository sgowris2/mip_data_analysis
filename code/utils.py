import pandas as pd
import numpy as np

import pandas as pd
import inspect
from html import escape


# Define a function to render Python objects to HTML dynamically
def render_dict_as_html(json_obj, title=None):
    """
    Dynamically renders an object as HTML.
    Includes attributes and dataframes, and formats them appropriately.
    """
    # Start the HTML structure
    html = f"<html><head><style>body {{ font-family: Arial, sans-serif; }} table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }} th {{ background-color: #f2f2f2; }} </style></head><body>"
    if title:
        html += "<h2>{}</h2>".format(title)

    # Dynamically fetch object's attributes and values
    for attribute, value in json_obj.items():

        attribute_text = attribute.replace('_', ' ')
        attribute_text = attribute_text.capitalize()

        # If the value is a pandas DataFrame, convert it to HTML
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            value_text = f"</br></br>{value.to_html(classes='dataframe', index=False)}</br>"

        # If the value is another object, render its attributes recursively
        elif hasattr(value, "__dict__"):
            value_text = render_dict_as_html(value)  # Recursive call to handle nested objects

        elif isinstance(value, list):
            value_text = ', '.join(value)

        # Otherwise, just display the value as text
        else:
            value_text = f"<a>{escape(str(value))}</a>"

        html += f"<p><strong>{escape(attribute_text)}:</strong> {value_text}</p>"

    # Close the HTML structure
    html += "</body></html>"

    return html


def merge_csvs(file1: str, file2: str, output_file: str):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df_out = pd.concat([df1, df2], ignore_index=True)
    df_out.to_csv(output_file, index=False)


def normalize_column_data(df: pd.DataFrame, columns: list):
    for col in columns:
        df[col] = df[col].replace('\'', '').replace('"', '')
    return df


def set_null_strings_to_none(df: pd.DataFrame):
    df = df.replace('Null', np.nan)
    df = df.replace('null', np.nan)
    return df


def normalize_column_names(df: pd.DataFrame):
    column_mapper = {x: _normalize_string(x) for x in df.columns.values}
    df.rename(columns=column_mapper, inplace=True)
    return df


def _normalize_string(s):
    s = s.strip().replace(' ', '_')
    s = str.lower(s)
    return s