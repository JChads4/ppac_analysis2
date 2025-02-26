python -c "
import panel as pn
import bokeh
import sys

print('Python version:', sys.version)
print('Panel version:', pn.__version__)
print('Bokeh version:', bokeh.__version__)
print('Panel extensions:', pn.extension.param.loaded)
"