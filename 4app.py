import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load data
df = pd.read_csv("gaming_laptops2_cleaned.csv")

# Data preprocessing - Better brand extraction

df['RAM_GB'] = pd.to_numeric(df['RAM'].str.extract(r'(\d+)')[0], errors='coerce').fillna(8).astype(int)
df['Storage_GB'] = pd.to_numeric(df['Storage'].str.extract(r'(\d+)')[0], errors='coerce').fillna(512).astype(int)


app = Dash(__name__, suppress_callback_exceptions=True)

# Color scheme (kept for python logic if needed, but mostly moved to CSS)
colors = {
    'background': '#0f0f23',
    'card_bg': '#1a1a2e',
    'text': '#ffffff',
    'accent': '#8b5cf6',
    'secondary': '#3b82f6',
    'success': '#10b981',
    'warning': '#f59e0b'
}

app.layout = html.Div(className='dashboard-container', children=[
    
    # Header
    html.Div(className='header-section', children=[
        html.H1('🎮 Gaming Laptops Analytics Dashboard'),
        html.P('Comprehensive EDA for Machine Learning Model Development')
    ]),
    
    # Key Metrics Row
    html.Div(className='metrics-grid', children=[
        html.Div(className='metric-card', style={'borderColor': colors['accent']}, children=[
            html.H3('Total Laptops', style={'color': colors['accent']}),
            html.H2(f"{len(df)}")
        ]),
        html.Div(className='metric-card', style={'borderColor': colors['secondary']}, children=[
            html.H3('Average Price', style={'color': colors['secondary']}),
            html.H2(f"{df['Price'].mean():.0f} TND")
        ]),
        html.Div(className='metric-card', style={'borderColor': colors['success']}, children=[
            html.H3('Avg RAM', style={'color': colors['success']}),
            html.H2(f"{df['RAM_GB'].mean():.1f} GB")
        ]),
        html.Div(className='metric-card', style={'borderColor': colors['warning']}, children=[
            html.H3('Unique Brands', style={'color': colors['warning']}),
            html.H2(f"{df['Brand'].nunique()}")
        ])
    ]),
    
    # Filters Section
    html.Div(className='card-auto', style={'marginBottom': '30px'}, children=[
        html.H3('🔍 Interactive Filters'),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))', 'gap': '20px'}, children=[
            html.Div(children=[
                html.Label('Select GPU:', style={'marginBottom': '8px', 'display': 'block', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='gpu-filter',
                    options=[{'label': 'All GPUs', 'value': 'All'}] + [{'label': gpu, 'value': gpu} for gpu in sorted(df['GPU'].unique())],
                    value='All',
                    clearable=False
                )
            ]),
            html.Div(children=[
                html.Label('Select Brand:', style={'marginBottom': '8px', 'display': 'block', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='brand-filter',
                    options=[{'label': 'All Brands', 'value': 'All'}] + [{'label': brand, 'value': brand} for brand in sorted(df['Brand'].unique())],
                    value='All',
                    clearable=False
                )
            ]),
            html.Div(children=[
                html.Label('Price Range (TND):', style={'marginBottom': '8px', 'display': 'block', 'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='price-range',
                    min=int(df['Price'].min()),
                    max=int(df['Price'].max()),
                    step=50,
                    value=[int(df['Price'].min()), int(df['Price'].max())],
                    marks={
                        int(df['Price'].min()): {'label': f"{int(df['Price'].min())}", 'style': {'color': colors['text']}}, 
                        int(df['Price'].max()): {'label': f"{int(df['Price'].max())}", 'style': {'color': colors['text']}}
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    ]),
    
    # Charts Grid - Row 1
    html.Div(className='charts-grid-2', children=[
        html.Div(className='card', children=[
            html.H3('💰 Price Distribution'),
            dcc.Graph(id='price-dist', config={'displayModeBar': False}, style={'flexGrow': 1})
        ]),
        html.Div(className='card', children=[
            html.H3('📊 RAM vs Price Analysis'),
            dcc.Graph(id='ram-price-scatter', config={'displayModeBar': False}, style={'flexGrow': 1})
        ])
    ]),
    
    # Charts Grid - Row 2
    html.Div(className='charts-grid-2', children=[
        html.Div(className='card', children=[
            html.H3('🎮 GPU Distribution'),
            dcc.Graph(id='gpu-dist', config={'displayModeBar': False}, style={'flexGrow': 1})
        ]),
        html.Div(className='card', children=[
            html.H3('🏷️ Brand Market Share'),
            dcc.Graph(id='brand-pie', config={'displayModeBar': False}, style={'flexGrow': 1})
        ])
    ]),
    
    # Charts Grid - Row 3
    html.Div(className='charts-grid-2', children=[
        html.Div(className='card', children=[
            html.H3('🖥️ Top 10 CPUs'),
            dcc.Graph(id='cpu-dist', config={'displayModeBar': False}, style={'flexGrow': 1})
        ]),
        html.Div(className='card', children=[
            html.H3('💾 Storage Distribution'),
            dcc.Graph(id='storage-dist', config={'displayModeBar': False}, style={'flexGrow': 1})
        ])
    ]),
    
    # Full Width Charts
    html.Div(className='charts-grid-full', children=[
        html.Div(className='card', children=[
            html.H3('📈 Average Price by Brand & GPU'),
            dcc.Graph(id='brand-gpu-price', config={'displayModeBar': False}, style={'flexGrow': 1})
        ])
    ]),
    
    # Correlation Heatmap
    html.Div(className='charts-grid-full', children=[
        html.Div(className='card', children=[
            html.H3('🔥 Feature Correlation Matrix'),
            dcc.Graph(id='correlation-heatmap', config={'displayModeBar': False}, style={'flexGrow': 1})
        ])
    ]),
    
    # Warranty & Color Analysis
    html.Div(className='charts-grid-2', children=[
        html.Div(className='card', children=[
            html.H3('🛡️ Warranty Distribution'),
            dcc.Graph(id='warranty-dist', config={'displayModeBar': False}, style={'flexGrow': 1})
        ]),
        html.Div(className='card', children=[
            html.H3('🎨 Color Preferences'),
            dcc.Graph(id='color-dist', config={'displayModeBar': False}, style={'flexGrow': 1})
        ])
    ])
])

# Callbacks
@app.callback(
    [Output('price-dist', 'figure'),
     Output('ram-price-scatter', 'figure'),
     Output('gpu-dist', 'figure'),
     Output('brand-pie', 'figure'),
     Output('cpu-dist', 'figure'),
     Output('storage-dist', 'figure'),
     Output('brand-gpu-price', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('warranty-dist', 'figure'),
     Output('color-dist', 'figure')],
    [Input('gpu-filter', 'value'),
     Input('brand-filter', 'value'),
     Input('price-range', 'value')]
)
def update_charts(gpu_filter, brand_filter, price_range):
    try:
        # Filter data
        filtered_df = df.copy()
        
        if gpu_filter != 'All':
            filtered_df = filtered_df[filtered_df['GPU'] == gpu_filter]
        
        if brand_filter != 'All':
            filtered_df = filtered_df[filtered_df['Brand'] == brand_filter]
        
        filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & 
                                  (filtered_df['Price'] <= price_range[1])]
        
        if filtered_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                plot_bgcolor=colors['card_bg'],
                paper_bgcolor=colors['card_bg'],
                font={'color': colors['text']}
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        # Common layout settings
        base_layout = {
            'plot_bgcolor': colors['card_bg'],
            'paper_bgcolor': colors['card_bg'],
            'font': {'color': colors['text'], 'size': 12},
        }
        
        # Helper to apply standard styling
        def apply_style(fig, margin=None):
            if margin is None:
                margin = {'l': 60, 'r': 30, 't': 30, 'b': 60}
            fig.update_layout(**base_layout, margin=margin)
            fig.update_xaxes(gridcolor='#374151')
            fig.update_yaxes(gridcolor='#374151')

        # 1. Price Distribution
        fig_price = px.histogram(filtered_df, x='Price', nbins=25, 
                                 color_discrete_sequence=[colors['accent']])
        apply_style(fig_price)
        fig_price.update_layout(
            showlegend=False,
            xaxis_title='Price (TND)',
            yaxis_title='Count',
            bargap=0.1
        )
        
        # 2. RAM vs Price Scatter
        fig_scatter = px.scatter(filtered_df, x='RAM_GB', y='Price', 
                                color='GPU', size='Storage_GB',
                                hover_data=['Name', 'CPU', 'Brand'])
        apply_style(fig_scatter)
        fig_scatter.update_layout(xaxis_title='RAM (GB)', yaxis_title='Price (TND)')
        
        # 3. GPU Distribution
        gpu_counts = filtered_df['GPU'].value_counts().reset_index()
        gpu_counts.columns = ['GPU', 'Count']
        fig_gpu = px.bar(gpu_counts, x='GPU', y='Count',
                         color='Count',
                         color_continuous_scale='Viridis')
        apply_style(fig_gpu)
        fig_gpu.update_layout(
            showlegend=False,
            xaxis_title='GPU Model',
            yaxis_title='Count'
        )
        fig_gpu.update_xaxes(tickangle=-45)
        
        # 4. Brand Pie Chart
        brand_counts = filtered_df['Brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig_brand_pie = px.pie(brand_counts, values='Count', names='Brand', hole=0.4)
        apply_style(fig_brand_pie, margin={'l': 20, 'r': 20, 't': 30, 'b': 20})
        
        # 5. CPU Distribution (Top 10)
        cpu_counts = filtered_df['CPU'].value_counts().head(10).reset_index()
        cpu_counts.columns = ['CPU', 'Count']
        fig_cpu = px.bar(cpu_counts, y='CPU', x='Count',
                         orientation='h',
                         color='Count',
                         color_continuous_scale='Blues')
        apply_style(fig_cpu, margin={'l': 180, 'r': 30, 't': 30, 'b': 60})
        fig_cpu.update_layout(
            showlegend=False,
            yaxis_title='CPU Model',
            xaxis_title='Count'
        )
        
        # 6. Storage Distribution
        storage_counts = filtered_df['Storage_GB'].value_counts().sort_index().reset_index()
        storage_counts.columns = ['Storage (GB)', 'Count']
        fig_storage = px.bar(storage_counts, x='Storage (GB)', y='Count',
                            color='Count',
                            color_continuous_scale='Greens')
        apply_style(fig_storage)
        fig_storage.update_layout(
            showlegend=False,
            xaxis_title='Storage (GB)',
            yaxis_title='Count'
        )
        
        # 7. Brand & GPU Average Price
        brand_gpu_avg = filtered_df.groupby(['Brand', 'GPU'])['Price'].mean().reset_index()
        brand_gpu_avg = brand_gpu_avg.sort_values('Price', ascending=False).head(20)
        fig_brand_gpu = px.bar(brand_gpu_avg, x='Brand', y='Price', color='GPU',
                              barmode='group')
        apply_style(fig_brand_gpu, margin={'l': 60, 'r': 30, 't': 30, 'b': 120})
        fig_brand_gpu.update_layout(
            xaxis_title='Brand',
            yaxis_title='Average Price (TND)',
            height=400
        )
        fig_brand_gpu.update_xaxes(tickangle=-45)
        
        # 8. Correlation Heatmap
        corr_features = filtered_df[['Price', 'RAM_GB', 'Storage_GB', 'Garentie']].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_features.values,
            x=corr_features.columns,
            y=corr_features.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_features.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar={'title': 'Correlation'}
        ))
        fig_corr.update_layout(
            **base_layout,
            height=500,
            margin={'l': 100, 'r': 50, 't': 50, 'b': 100}
        )
        
        # 9. Warranty Distribution
        warranty_counts = filtered_df['Garentie'].value_counts().reset_index()
        warranty_counts.columns = ['Warranty', 'Count']
        warranty_counts['Warranty'] = warranty_counts['Warranty'].astype(str) + ' Year(s)'
        fig_warranty = px.pie(warranty_counts, values='Count', names='Warranty',
                             color_discrete_sequence=px.colors.sequential.Sunset)
        apply_style(fig_warranty, margin={'l': 20, 'r': 20, 't': 30, 'b': 20})
        
        # 10. Color Distribution
        color_counts = filtered_df['Color'].value_counts().head(10).reset_index()
        color_counts.columns = ['Color', 'Count']
        fig_color = px.bar(color_counts, x='Color', y='Count',
                          color='Count',
                          color_continuous_scale='Rainbow')
        apply_style(fig_color)
        fig_color.update_layout(
            showlegend=False,
            xaxis_title='Color',
            yaxis_title='Count'
        )
        fig_color.update_xaxes(tickangle=-45)
        
        return fig_price, fig_scatter, fig_gpu, fig_brand_pie, fig_cpu, fig_storage, fig_brand_gpu, fig_corr, fig_warranty, fig_color
        
    except Exception as e:
        print(f"ERROR in callback: {e}")
        import traceback
        traceback.print_exc()
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}", font={'color': 'red'})
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

if __name__ == "__main__":
    app.run(debug=True)