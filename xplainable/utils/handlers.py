

def add_thousands_separator(var):
    if isinstance(var, (float)):
        return '{:,.2f}'.format(var)
    elif isinstance(var, (int)):
        return '{:,.0f}'.format(var)
    else:
        return var