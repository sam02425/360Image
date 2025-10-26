#!/usr/bin/env python3
"""
Real-time Interactive Dashboard for Multi-View Experiment Monitoring
Monitors GPU usage, training progress, completed runs, and remaining experiments
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import psutil
import GPUtil
import glob
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue

class ExperimentMonitor:
    """Real-time experiment monitoring system"""
    
    def __init__(self, results_dir="results/training_runs", refresh_interval=5):
        self.results_dir = Path(results_dir)
        self.refresh_interval = refresh_interval
        self.data_queue = queue.Queue()
        
        # Initialize data storage
        self.gpu_data = []
        self.system_data = []
        self.experiment_status = {}
        self.current_metrics = {}
        
        # Expected experiments (5 runs × 2 architectures × 4 configs = 40)
        self.total_experiments = 40
        self.expected_runs = self.generate_expected_runs()
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.background_monitor, daemon=True)
        self.monitor_thread.start()
    
    def generate_expected_runs(self):
        """Generate list of all expected experiment runs"""
        expected = []
        architectures = ['yolov8n', 'yolov11n']
        view_configs = ['dual', 'quad', 'octal', 'full']
        
        for arch in architectures:
            for config in view_configs:
                for run_idx in range(5):  # 5 runs each
                    expected.append(f"{arch}_{config}_run{run_idx}_fold0")
        
        return expected
    
    def get_system_metrics(self):
        """Get current system and GPU metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'gpu_utilization': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'gpu_memory_percent': 0,
            'gpu_temperature': 0
        }
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                })
        except:
            pass
        
        return metrics
    
    def scan_experiment_status(self):
        """Scan current experiment status"""
        status = {
            'completed': [],
            'running': [],
            'failed': [],
            'pending': []
        }
        
        # Check each expected run
        for run_name in self.expected_runs:
            run_dir = self.results_dir / run_name
            
            if run_dir.exists():
                results_csv = run_dir / 'results.csv'
                weights_dir = run_dir / 'weights'
                
                if (weights_dir / 'best.pt').exists():
                    status['completed'].append(run_name)
                elif results_csv.exists():
                    # Check if currently training (recent file modification)
                    last_modified = results_csv.stat().st_mtime
                    if time.time() - last_modified < 300:  # Modified in last 5 minutes
                        status['running'].append(run_name)
                    else:
                        status['failed'].append(run_name)
                else:
                    status['failed'].append(run_name)
            else:
                status['pending'].append(run_name)
        
        return status
    
    def get_current_training_metrics(self):
        """Get metrics from currently running experiments"""
        current_metrics = {}
        
        # Find running experiments
        for run_dir in self.results_dir.glob("*_run*_fold*"):
            results_csv = run_dir / 'results.csv'
            
            if results_csv.exists():
                last_modified = results_csv.stat().st_mtime
                if time.time() - last_modified < 300:  # Recent activity
                    try:
                        df = pd.read_csv(results_csv)
                        if len(df) > 0:
                            latest = df.iloc[-1]
                            current_metrics[run_dir.name] = {
                                'epoch': len(df),
                                'map50': latest.get('metrics/mAP50(B)', 0),
                                'precision': latest.get('metrics/precision(B)', 0),
                                'recall': latest.get('metrics/recall(B)', 0),
                                'box_loss': latest.get('train/box_loss', 0),
                                'cls_loss': latest.get('train/cls_loss', 0),
                                'training_time': latest.get('time', 0) / 60  # Convert to minutes
                            }
                    except:
                        pass
        
        return current_metrics
    
    def background_monitor(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                experiment_status = self.scan_experiment_status()
                current_metrics = self.get_current_training_metrics()
                
                # Store data (keep last 100 points for plotting)
                self.gpu_data.append(system_metrics)
                if len(self.gpu_data) > 100:
                    self.gpu_data.pop(0)
                
                self.experiment_status = experiment_status
                self.current_metrics = current_metrics
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.refresh_interval)

# Initialize monitor
monitor = ExperimentMonitor()

# Create Dash app
app = dash.Dash(__name__)
app.title = "Multi-View Experiment Monitor"

# Define layout
app.layout = html.Div([
    html.H1("Multi-View YOLOv8/YOLOv11 Experiment Monitor", 
           style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Auto-refresh component
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    # Overview cards
    html.Div(id='overview-cards', children=[
        html.Div([
            html.H3("Experiment Progress"),
            html.Div(id='progress-summary')
        ], className='card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%'}),
        
        html.Div([
            html.H3("GPU Status"),
            html.Div(id='gpu-summary')
        ], className='card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%'}),
        
        html.Div([
            html.H3("System Status"),
            html.Div(id='system-summary')
        ], className='card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%'}),
        
        html.Div([
            html.H3("Time Estimate"),
            html.Div(id='time-estimate')
        ], className='card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%'}),
    ], style={'marginBottom': 30}),
    
    # Charts row
    html.Div([
        html.Div([
            dcc.Graph(id='gpu-utilization-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='memory-usage-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
    ], style={'marginBottom': 30}),
    
    # Current training progress
    html.Div([
        html.H2("Current Training Progress"),
        html.Div(id='current-training-table')
    ], style={'marginBottom': 30}),
    
    # Experiment status tables
    html.Div([
        html.Div([
            html.H3("Completed Experiments"),
            html.Div(id='completed-table')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.H3("Pending Experiments"),
            html.Div(id='pending-table')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
    ])
])

# Callback for updating overview cards
@app.callback(
    [Output('progress-summary', 'children'),
     Output('gpu-summary', 'children'),
     Output('system-summary', 'children'),
     Output('time-estimate', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_overview(n):
    # Progress summary
    status = monitor.experiment_status
    completed = len(status.get('completed', []))
    running = len(status.get('running', []))
    failed = len(status.get('failed', []))
    pending = len(status.get('pending', []))
    total = monitor.total_experiments
    
    progress_percent = (completed / total) * 100 if total > 0 else 0
    
    progress_summary = [
        html.P(f"Completed: {completed}/{total}"),
        html.P(f"Running: {running}"),
        html.P(f"Failed: {failed}"),
        html.P(f"Progress: {progress_percent:.1f}%")
    ]
    
    # GPU summary
    if monitor.gpu_data:
        latest_gpu = monitor.gpu_data[-1]
        gpu_summary = [
            html.P(f"Utilization: {latest_gpu['gpu_utilization']:.1f}%"),
            html.P(f"Memory: {latest_gpu['gpu_memory_percent']:.1f}%"),
            html.P(f"Temp: {latest_gpu['gpu_temperature']:.0f}°C")
        ]
    else:
        gpu_summary = [html.P("No data")]
    
    # System summary
    if monitor.gpu_data:
        latest_sys = monitor.gpu_data[-1]
        system_summary = [
            html.P(f"CPU: {latest_sys['cpu_percent']:.1f}%"),
            html.P(f"RAM: {latest_sys['memory_percent']:.1f}%")
        ]
    else:
        system_summary = [html.P("No data")]
    
    # Time estimate
    if completed > 0 and running > 0:
        avg_time_per_experiment = 120  # Rough estimate in minutes
        remaining_time = (pending + running) * avg_time_per_experiment
        hours = remaining_time // 60
        minutes = remaining_time % 60
        time_estimate = [
            html.P(f"Est. remaining:"),
            html.P(f"{hours}h {minutes}m")
        ]
    else:
        time_estimate = [html.P("Calculating...")]
    
    return progress_summary, gpu_summary, system_summary, time_estimate

# Callback for GPU utilization chart
@app.callback(
    Output('gpu-utilization-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_gpu_chart(n):
    if not monitor.gpu_data:
        return go.Figure()
    
    df = pd.DataFrame(monitor.gpu_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['gpu_utilization'],
        mode='lines',
        name='GPU Utilization (%)',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        title="GPU Utilization Over Time",
        xaxis_title="Time",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Callback for memory usage chart
@app.callback(
    Output('memory-usage-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_memory_chart(n):
    if not monitor.gpu_data:
        return go.Figure()
    
    df = pd.DataFrame(monitor.gpu_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['gpu_memory_percent'],
        mode='lines',
        name='GPU Memory (%)',
        line=dict(color='#ff7f0e')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['memory_percent'],
        mode='lines',
        name='System Memory (%)',
        line=dict(color='#2ca02c')
    ))
    
    fig.update_layout(
        title="Memory Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Memory Usage (%)",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Callback for current training table
@app.callback(
    Output('current-training-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_current_training(n):
    if not monitor.current_metrics:
        return html.P("No training currently running")
    
    data = []
    for exp_name, metrics in monitor.current_metrics.items():
        data.append({
            'Experiment': exp_name,
            'Epoch': metrics['epoch'],
            'mAP@0.5': f"{metrics['map50']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'Box Loss': f"{metrics['box_loss']:.4f}",
            'Time (min)': f"{metrics['training_time']:.1f}"
        })
    
    return dash_table.DataTable(
        data=data,
        columns=[{"name": i, "id": i} for i in data[0].keys()] if data else [],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    )

# Callback for completed experiments table
@app.callback(
    Output('completed-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_completed_table(n):
    completed = monitor.experiment_status.get('completed', [])
    
    if not completed:
        return html.P("No experiments completed yet")
    
    # Get final results for completed experiments
    data = []
    for exp_name in completed[-10:]:  # Show last 10 completed
        exp_dir = monitor.results_dir / exp_name
        results_csv = exp_dir / 'results.csv'
        
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                if len(df) > 0:
                    final = df.iloc[-1]
                    data.append({
                        'Experiment': exp_name,
                        'Final mAP@0.5': f"{final.get('metrics/mAP50(B)', 0):.3f}",
                        'Epochs': len(df),
                        'Time (min)': f"{final.get('time', 0)/60:.1f}"
                    })
            except:
                pass
    
    if not data:
        return html.P("No completed data available")
    
    return dash_table.DataTable(
        data=data,
        columns=[{"name": i, "id": i} for i in data[0].keys()],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'rgb(200, 255, 200)', 'fontWeight': 'bold'}
    )

# Callback for pending experiments table
@app.callback(
    Output('pending-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_pending_table(n):
    pending = monitor.experiment_status.get('pending', [])
    
    if not pending:
        return html.P("No pending experiments")
    
    data = [{'Pending Experiment': exp} for exp in pending[:10]]  # Show first 10
    
    return dash_table.DataTable(
        data=data,
        columns=[{"name": "Pending Experiment", "id": "Pending Experiment"}],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'rgb(255, 200, 200)', 'fontWeight': 'bold'}
    )

# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .card h3 {
                margin-top: 0;
                color: #333;
            }
            .card p {
                margin: 5px 0;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("Starting Multi-View Experiment Monitor...")
    print("Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        monitor.monitoring_active = False
        print("\nShutting down monitor...")