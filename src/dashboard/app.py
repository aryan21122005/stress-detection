import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from pathlib import Path
from typing import List, Dict, Optional

# Initialize the Dash app
app = Dash(__name__)

def list_sessions(sessions_dir: str = "sessions") -> List[Dict]:
    """List all available sessions with their metadata."""
    sessions = []
    for file in Path(sessions_dir).glob("*.csv"):
        try:
            df = pd.read_csv(file)
            if not df.empty:
                sessions.append({
                    'id': file.stem,
                    'path': str(file),
                    'start_time': df['datetime'].min(),
                    'duration': (pd.to_datetime(df['datetime']).max() - 
                               pd.to_datetime(df['datetime']).min()).total_seconds() / 60,  # in minutes
                    'avg_attention': df['attention_score'].mean(),
                    'focus_percentage': (df['attention_score'] > 0.7).mean() * 100,
                    'dominant_emotion': df['emotion'].mode()[0] if not df['emotion'].empty else 'unknown'
                })
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return sessions

# Layout
app.layout = html.Div([
    html.H1("Attention & Focus Analytics Dashboard", style={'textAlign': 'center'}),
    
    # Session Selection
    html.Div([
        html.H3("Select Session"),
        dcc.Dropdown(
            id='session-selector',
            options=[],
            placeholder="Select a session to analyze"
        ),
    ], style={'width': '80%', 'margin': '20px auto'}),
    
    # Summary Cards
    html.Div(id='summary-cards', style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'margin': '20px 0'
    }),
    
    # Main Graphs
    html.Div([
        dcc.Graph(id='attention-timeline'),
        dcc.Graph(id='emotion-distribution'),
        dcc.Graph(id='attention-heatmap')
    ], style={'width': '90%', 'margin': '0 auto'}),
    
    # Hidden div to store session data
    dcc.Store(id='session-data')
])

@app.callback(
    Output('session-selector', 'options'),
    Input('session-selector', 'id')
)
def update_session_options(_):
    sessions = list_sessions()
    return [{'label': f"{s['id']} ({s['start_time']})", 'value': s['path']} for s in sessions]

@app.callback(
    Output('session-data', 'data'),
    Input('session-selector', 'value')
)
def load_session_data(selected_session):
    if not selected_session:
        return {}
    try:
        df = pd.read_csv(selected_session)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading session data: {e}")
        return {}

@app.callback(
    Output('summary-cards', 'children'),
    Input('session-data', 'data')
)
def update_summary_cards(data):
    if not data:
        return []
    
    df = pd.DataFrame(data)
    if df.empty:
        return []
    
    avg_attention = df['attention_score'].mean() * 100
    focus_percentage = (df['attention_score'] > 0.7).mean() * 100
    dominant_emotion = df['emotion'].mode()[0] if not df['emotion'].empty else 'N/A'
    avg_stress = df['stress_level'].mean() * 100
    
    cards = [
        html.Div([
            html.H3("Avg. Attention"),
            html.Div(f"{avg_attention:.1f}%", style={'fontSize': '24px', 'color': '#2ecc71'})
        ], className="card"),
        
        html.Div([
            html.H3("Focus %"),
            html.Div(f"{focus_percentage:.1f}%", style={'fontSize': '24px', 'color': '#3498db'})
        ], className="card"),
        
        html.Div([
            html.H3("Dominant Emotion"),
            html.Div(dominant_emotion, style={'fontSize': '24px', 'color': '#9b59b6'})
        ], className="card"),
        
        html.Div([
            html.H3("Avg. Stress"),
            html.Div(f"{avg_stress:.1f}%", style={'fontSize': '24px', 'color': '#e74c3c'})
        ], className="card")
    ]
    
    return cards

@app.callback(
    Output('attention-timeline', 'figure'),
    Input('session-data', 'data')
)
def update_attention_timeline(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    
    fig = px.line(
        df, 
        x='datetime', 
        y='attention_score',
        title='Attention Score Over Time',
        labels={'attention_score': 'Attention Score', 'datetime': 'Time'}
    )
    
    # Add a horizontal line at 0.7 for focus threshold
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                 annotation_text="Focus Threshold", 
                 annotation_position="bottom right")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Attention Score",
        yaxis_range=[0, 1.1],
        hovermode="x unified"
    )
    
    return fig

@app.callback(
    Output('emotion-distribution', 'figure'),
    Input('session-data', 'data')
)
def update_emotion_distribution(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    
    fig = px.pie(
        emotion_counts,
        names='emotion',
        values='count',
        title='Emotion Distribution',
        hole=0.3
    )
    
    return fig

@app.callback(
    Output('attention-heatmap', 'figure'),
    Input('session-data', 'data')
)
def update_attention_heatmap(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    
    # Extract hour and minute from datetime
    df['time'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['time'].dt.hour
    df['minute_bin'] = (df['time'].dt.minute // 5) * 5  # Bin by 5 minutes
    
    # Pivot for heatmap
    heatmap_data = df.pivot_table(
        index='hour',
        columns='minute_bin',
        values='attention_score',
        aggfunc='mean',
        fill_value=0
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Minute", y="Hour", color="Attention"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title="Attention Heatmap by Time of Day",
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    
    fig.update_xaxes(side="bottom")
    
    return fig

# Add some basic CSS
app.layout.children.append(html.Div(className='styles', children=[
    dcc.Markdown("""
    <style>
        .card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 20%;
            min-width: 150px;
        }
        
        .card h3 {
            margin: 0 0 10px 0;
            color: #555;
            font-size: 16px;
        }
        
        .graph-container {
            margin: 30px 0;
        }
        
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        #main-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """)
]))

if __name__ == '__main__':
    # Create sessions directory if it doesn't exist
    os.makedirs("sessions", exist_ok=True)
    
    # Run the app
    app.run_server(debug=True, port=8050)
