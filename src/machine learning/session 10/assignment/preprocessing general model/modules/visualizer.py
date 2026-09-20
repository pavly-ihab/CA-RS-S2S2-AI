# modules/visualizer.py
"""
Step 12: Universal Data Visualization (Plotly Dashboards & Distributions)
Dynamically generates:
1. A Single Unified Master Interactive Dashboard (dashboard.html) combining all charts
2. Individual interactive HTML figures
3. Automatically opens the dashboard in your default browser if requested.
"""
from pathlib import Path
from typing import List, Optional, Dict
import webbrowser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_unified_html_dashboard(
    figures_dict: Dict[str, go.Figure],
    dataset_name: str,
    output_path: Path,
    summary_stats: Optional[Dict[str, str]] = None
) -> Path:
    """
    Compiles multiple Plotly figures into a single unified dashboard HTML file
    with modern styling, sticky navigation, and responsive cards.
    """
    fig_divs = []
    nav_links = []

    for idx, (title, fig) in enumerate(figures_dict.items(), 1):
        section_id = f"section_{idx}"
        nav_links.append(f'<a href="#{section_id}" class="nav-item">{title}</a>')
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
        fig_divs.append(f"""
        <section id="{section_id}" class="chart-card">
            <div class="card-header">
                <h2>{title}</h2>
            </div>
            <div class="chart-container">
                {fig_html}
            </div>
        </section>
        """)

    stats_html = ""
    if summary_stats:
        stat_cards = "".join([
            f'<div class="stat-card"><span class="stat-title">{k}</span><span class="stat-val">{v}</span></div>'
            for k, v in summary_stats.items()
        ])
        stats_html = f'<div class="stats-row">{stat_cards}</div>'

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} — Preprocessing Visual Dashboard</title>
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-main: #c9d1d9;
            --text-heading: #f0f6fc;
            --accent: #58a6ff;
            --accent-glow: rgba(88, 166, 255, 0.15);
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.5;
            padding-bottom: 60px;
        }}
        header {{
            background: rgba(22, 27, 34, 0.85);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 1000;
            padding: 14px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .logo {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            background: var(--accent-glow);
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
        }}
        nav {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .nav-item {{
            color: var(--text-main);
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: 6px;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}
        .nav-item:hover {{
            background: var(--card-bg);
            color: var(--accent);
            border-color: var(--border-color);
        }}
        main {{
            max-width: 1280px;
            margin: 30px auto;
            padding: 0 20px;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
        }}
        .stat-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #8b949e;
            font-weight: 600;
        }}
        .stat-val {{
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-heading);
            margin-top: 4px;
        }}
        .chart-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-bottom: 30px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }}
        .card-header {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            background: #1c2128;
        }}
        .card-header h2 {{
            font-size: 1.1rem;
            color: var(--text-heading);
            font-weight: 600;
        }}
        .chart-container {{
            padding: 10px;
            min-height: 480px;
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo">
            📊 {dataset_name}
            <span class="badge">Universal Preprocessing Engine</span>
        </div>
        <nav>
            {"".join(nav_links)}
        </nav>
    </header>
    <main>
        {stats_html}
        {"".join(fig_divs)}
    </main>
</body>
</html>"""

    output_path.write_text(html_template, encoding="utf-8")
    return output_path


def generate_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: Optional[str] = None,
    task_type: str = "classification",
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    auto_open: bool = False
) -> Path:
    """
    Step 12: Generates individual figures AND a single unified master dashboard.html.
    """
    if df is None or df.empty:
        print("[Warning] Empty DataFrame provided for visualization.")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figures_dict = {}

    plot_df = df.copy()

    if cat_cols is None:
        cat_cols = [
            c for c in plot_df.columns
            if c != target_col and (
                str(plot_df[c].dtype) == "category" or
                plot_df[c].dtype == object or
                plot_df[c].nunique() <= 15
            )
        ]
    if num_cols is None:
        num_cols = [
            c for c in plot_df.select_dtypes(include=["number"]).columns
            if c != target_col and plot_df[c].nunique() > 10
        ]

    print("\n--- Generating Universal Plotly Visualizations ---")

    # 1. Target Distribution
    if target_col and target_col in plot_df.columns:
        if task_type.lower() == "classification":
            target_series = plot_df[target_col].astype(str)
            pie_fig = px.pie(
                names=target_series,
                title=f"<b>Target '{target_col}' Distribution</b>",
                hole=0.6,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            pie_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                title_x=0.5,
                annotations=[dict(text=f"{target_col}", x=0.5, y=0.5, font_size=18, showarrow=False)]
            )
            figures_dict["Target Distribution"] = pie_fig
            pie_fig.write_html(str(out_path / "1_target_distribution.html"))
        else:
            reg_fig = px.histogram(
                plot_df,
                x=target_col,
                marginal="box",
                title=f"<b>Target '{target_col}' Continuous Distribution (Regression)</b>",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            reg_fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22", title_x=0.5)
            figures_dict["Target Distribution"] = reg_fig
            reg_fig.write_html(str(out_path / "1_target_distribution.html"))

    # 2. Categorical Subplot Dashboard
    active_cats = [c for c in cat_cols if c in plot_df.columns][:9]
    if active_cats:
        n_cats = len(active_cats)
        cols = 3 if n_cats >= 3 else n_cats
        rows = (n_cats + cols - 1) // cols
        subplot_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"<b>{c}</b>" for c in active_cats]
        )
        for i, col in enumerate(active_cats):
            r = (i // cols) + 1
            c = (i % cols) + 1
            counts = plot_df[col].astype(str).value_counts().head(10)
            subplot_fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker_color=px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)],
                    name=col,
                    showlegend=False
                ),
                row=r, col=c
            )
        subplot_fig.update_layout(
            title_text="<b>Categorical Features Subplot Dashboard</b>",
            title_x=0.5,
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            height=300 * rows
        )
        figures_dict["Categorical Dashboard"] = subplot_fig
        subplot_fig.write_html(str(out_path / "2_categorical_dashboard.html"))

    # 3. Outlier Boxplots
    active_nums = [c for c in num_cols if c in plot_df.columns]
    for num_col in active_nums:
        has_cat_target = (target_col and target_col in plot_df.columns and task_type.lower() == "classification")
        box_fig = px.box(
            plot_df,
            y=num_col,
            x=target_col if has_cat_target else None,
            color=target_col if has_cat_target else None,
            points="all",
            title=f"<b>Box Plot: {num_col} (Outlier & Distribution Analysis)</b>",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        box_fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22", title_x=0.5)
        figures_dict[f"Outlier Box Plot ({num_col})"] = box_fig
        box_fig.write_html(str(out_path / f"3_box_{num_col.lower()}_outliers.html"))

    # 4. Continuous Feature Distributions
    for num_col in active_nums:
        has_cat_target = (target_col and target_col in plot_df.columns and task_type.lower() == "classification")
        dist_fig = px.histogram(
            plot_df,
            x=num_col,
            color=target_col if has_cat_target else None,
            marginal="box",
            barmode="overlay",
            opacity=0.7,
            title=f"<b>Distribution of {num_col}</b>",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        dist_fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22", title_x=0.5)
        figures_dict[f"Distribution ({num_col})"] = dist_fig
        dist_fig.write_html(str(out_path / f"4_dist_{num_col.lower()}.html"))

    # 5. Correlation Heatmap
    numeric_df = plot_df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().round(2)
        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="<b>Numerical Feature Correlation Matrix</b>"
        )
        heatmap_fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22", title_x=0.5)
        figures_dict["Correlation Heatmap"] = heatmap_fig
        heatmap_fig.write_html(str(out_path / "5_correlation_heatmap.html"))

    # Build Master Unified Dashboard
    dashboard_path = out_path / "dashboard.html"
    summary_stats = {
        "Total Records": f"{plot_df.shape[0]:,}",
        "Features Count": str(plot_df.shape[1]),
        "Target Feature": str(target_col),
        "Problem Type": task_type.capitalize()
    }
    build_unified_html_dashboard(
        figures_dict=figures_dict,
        dataset_name=out_path.name,
        output_path=dashboard_path,
        summary_stats=summary_stats
    )

    print(f"\n[Success] Unified Master Dashboard Generated: {dashboard_path.resolve()}")
    if auto_open:
        print(f"[Action] Opening {dashboard_path.name} in default browser...")
        webbrowser.open(dashboard_path.resolve().as_uri())

    return dashboard_path
