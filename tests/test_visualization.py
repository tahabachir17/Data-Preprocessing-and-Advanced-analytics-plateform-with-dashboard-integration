import pandas as pd

from src.visualization import dashboard as dashboard_module
from src.visualization.charts import DataVisualizer
from src.visualization.dashboard import Dashboard


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_time_series_suggestions_do_not_prioritize_violin():
    visualizer = DataVisualizer()
    df = pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=120, freq="h"),
            "value": range(120),
        }
    )

    suggestions = visualizer.get_plot_suggestions(df, "date_time", "value")

    assert "Time Series Plot" in suggestions
    assert "Line Plot" in suggestions
    assert "Violin Plot" not in suggestions


def test_violin_plot_buckets_high_cardinality_datetime_axis():
    visualizer = DataVisualizer()
    df = pd.DataFrame(
        {
            "date_time": pd.date_range("2026-01-01", periods=2000, freq="h"),
            "reading": [float(i % 50) for i in range(2000)],
        }
    )

    fig = visualizer.create_violin_plot(df, "date_time", "reading")
    rendered_groups = {value for trace in fig.data for value in getattr(trace, "x", []) if value is not None}

    assert len(rendered_groups) <= visualizer.max_distribution_groups
    assert len(rendered_groups) < df["date_time"].nunique()


def test_plot_type_selector_uses_reordered_options_index(monkeypatch):
    captured = {}
    dashboard = Dashboard(DataVisualizer())
    df = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

    monkeypatch.setattr(dashboard_module.st, "session_state", {"viz_selected_plot": "Violin Plot"}, raising=False)
    monkeypatch.setattr(dashboard_module.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard_module.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(dashboard_module.st, "columns", lambda count: [_DummyColumn() for _ in range(count)])

    def fake_selectbox(label, options, index=0, **kwargs):
        captured["options"] = options
        captured["index"] = index
        return options[index]

    monkeypatch.setattr(dashboard_module.st, "selectbox", fake_selectbox)

    selected = dashboard.render_plot_type_selector(df, "category", "value", key_prefix="viz")

    assert captured["options"][captured["index"]] == "Violin Plot"
    assert selected == "Violin Plot"
