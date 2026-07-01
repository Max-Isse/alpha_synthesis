import sys
import pytest

def pytest_configure(config):
    """
    Executes before tests initialize. Monkey-patches Plotly to bypass
    the deprecated vectorbt 'heatmapgl' schema property registration error.
    """
    try:
        import plotly.graph_objs.layout.template as template
        
        # Verify if the Data object exists and lacks 'heatmapgl'
        if hasattr(template, 'Data') and not hasattr(template.Data, 'heatmapgl'):
            # Inject a safe property alias directing to standard heatmap validation
            template.Data.heatmapgl = property(
                fget=lambda self: getattr(self, 'heatmap', None),
                fset=lambda self, val: setattr(self, 'heatmap', val)
            )
    except ImportError:
        pass # Plotly is not installed or configured in this pass