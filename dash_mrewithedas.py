import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import time
import json
import logging
import base64
import sys
import os
from dash.exceptions import PreventUpdate
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
import warnings

import mre  

# Add parent directory to sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import session management components (absolute imports)
try:
    from dash_session_manager import start_session_manager, get_session_manager
    from dash_session_components import create_session_components, setup_session_callbacks, register_long_running_process
    SESSION_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Session management not available: {e}")
    SESSION_MANAGEMENT_AVAILABLE = False
    # Define dummy functions to prevent errors
    def start_session_manager(): pass
    def get_session_manager(): return None
    def create_session_components(): return None, html.Div()
    def setup_session_callbacks(app): pass
    def register_long_running_process(session_id): pass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Start the global session manager
if SESSION_MANAGEMENT_AVAILABLE:
    start_session_manager()

# Global variables
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://bayes-interpret.com/Evidence/MREWithEDAsDash/assets/liquid-glass.css'  # Apple Liquid Glass CSS
    ],
    requests_pathname_prefix='/Evidence/MREWithEDAsDash/',
    suppress_callback_exceptions=True
)
server = app.server

# Safari Compatibility CSS Fix for Liquid Glass Effects
SAFARI_FIX_CSS = """
<style>
/* === SAFARI LIQUID GLASS COMPATIBILITY FIXES === */
/* Fixes for Safari 18 backdrop-filter + background-color bug */

/* Safari detection using CSS only */
@media not all and (min-resolution:.001dpcm) {
    @supports (-webkit-appearance:none) {
        
        /* Fix for main cards - separate background and blur */
        .card {
            background: transparent !important;
        }
        
        .card::before {
            background: rgba(255, 255, 255, 0.12) !important;
            -webkit-backdrop-filter: blur(15px) saturate(180%) !important;
            backdrop-filter: blur(15px) saturate(180%) !important;
        }
        
        /* Fix for buttons - use webkit prefix and avoid background conflicts */
        .btn {
            background: transparent !important;
            -webkit-backdrop-filter: blur(15px) !important;
            backdrop-filter: blur(15px) !important;
        }
        
        .btn::before {
            background: rgba(255, 255, 255, 0.12) !important;
        }
        
        /* Fix for containers - separate blur and background layers */
        #evidence-checkbox-container,
        #target-checkbox-container,
        #algorithm-checkbox-container {
            background: transparent !important;
            position: relative !important;
        }
        
        #evidence-checkbox-container::before,
        #target-checkbox-container::before,
        #algorithm-checkbox-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.08) !important;
            border-radius: inherit;
            z-index: -1;
            -webkit-backdrop-filter: blur(10px) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Fix for form controls */
        .form-control {
            background: rgba(255, 255, 255, 0.15) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Fix for upload card */
        .upload-card {
            background: rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Fix for notification container */
        #notification-container {
            background: rgba(255, 255, 255, 0.15) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            backdrop-filter: blur(15px) !important;
        }
    }
}

/* Fallback for very old Safari versions */
@supports not (backdrop-filter: blur(1px)) {
    .card {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    .btn {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    #evidence-checkbox-container,
    #target-checkbox-container,
    #algorithm-checkbox-container {
        background: rgba(255, 255, 255, 0.1) !important;
    }
}
</style>
"""

# IMPORTANT: We do NOT load asia.bif by default. Instead, we store None here:
default_model = None

# All algorithms can work with 1 or more target variables
# The requirement of 2+ variables was a performance optimization, not a technical limitation
algorithm_requirements = {
    'UMDAcat_mre2': 1,
    'DEA MRE': 1,
    'EBNA MRE': 1,
    'ES MRE': 1,
    'GA MRE': 1,
    'Hierarchical Beam Search': 1,
    'NSGA2 MRE': 1,
    'PSO MRE': 1,
    'Tabu MRE': 1
}

# Add this function after the algorithm_requirements dictionary and before the session components creation

def estimate_algorithm_time(algorithm, num_targets, num_evidence, num_nodes, pop_size, n_gen, max_steps, dead_iter):
    """
    Estimate the execution time for an algorithm based on computational complexity theory.
    All estimates are calibrated using Asia network (8 nodes) as baseline.
    Returns estimated time in seconds.
    """
    
    # === BASELINE TIMES (seconds) ===
    # Based on empirical measurements with Asia network (8 nodes, 1 target, 0 evidence)
    # Population=50, Generations/Steps=50-100, Dead_iter=10
    baseline_times = {
        'Hierarchical Beam Search': 0.8,  # O(b^d) - fastest, deterministic search
        'Tabu MRE': 1.2,                  # O(n*m) - local search with memory
        'PSO MRE': 1.8,                   # O(p*g*n) - particle swarm, fast convergence
        'DEA MRE': 2.4,                   # O(p*g*n) - differential evolution
        'ES MRE': 2.6,                    # O(p*g*n) - evolution strategy
        'GA MRE': 3.1,                    # O(p*g*n) - genetic algorithm with crossover overhead
        'UMDAcat_mre2': 3.8,              # O(p*g*n²) - builds probability distributions
        'EBNA MRE': 5.2,                  # O(p*g*n³) - builds Bayesian networks each generation
        'NSGA2 MRE': 4.7                  # O(p*g*n*log(n)) - multi-objective with sorting
    }
    
    if algorithm not in baseline_times:
        return 0.0
    
    baseline_time = baseline_times[algorithm]
    
    # === COMPUTATIONAL COMPLEXITY FACTORS ===
    
    # 1. NETWORK COMPLEXITY: O(2^n) for probabilistic inference
    # Each additional node exponentially increases complexity
    network_factor = 2 ** (max(0, num_nodes - 8) * 0.15)  # Exponential scaling, moderated
    
    # 2. TARGET COMPLEXITY: O(2^t) for solution space
    # Each target variable multiplies solution space exponentially
    target_factor = (2 ** num_targets) / 2  # Normalized to single target
    
    # 3. EVIDENCE IMPACT: Reduces search space but increases inference cost
    # Evidence reduces solution space but increases probabilistic queries
    if num_evidence == 0:
        evidence_factor = 1.0
    else:
        # Initial reduction for pruning search space, then cost for inference
        evidence_factor = max(0.3, 1.0 - num_evidence * 0.08) + num_evidence * 0.05
    
    # === ALGORITHM-SPECIFIC PARAMETER SCALING ===
    
    if algorithm in ['UMDAcat_mre2', 'EBNA MRE']:
        # EDAs: O(p * g * n²) or O(p * g * n³)
        # Population size has quadratic/cubic impact due to distribution learning
        pop_factor = (pop_size / 50) ** 1.5
        iter_factor = dead_iter / 10
        param_factor = pop_factor * iter_factor
        
    elif algorithm in ['GA MRE', 'ES MRE', 'DEA MRE', 'PSO MRE']:
        # Evolutionary algorithms: O(p * g * n)
        # Linear scaling with population and generations
        pop_factor = pop_size / 50
        step_factor = max_steps / 100
        param_factor = pop_factor * step_factor
        
    elif algorithm == 'NSGA2 MRE':
        # NSGA2: O(p * g * n * log(n)) due to non-dominated sorting
        pop_factor = pop_size / 50
        gen_factor = n_gen / 50
        sorting_overhead = 1 + 0.1 * pop_size / 50  # log(n) approximation
        param_factor = pop_factor * gen_factor * sorting_overhead
        
    elif algorithm == 'Tabu MRE':
        # Tabu Search: O(steps * n)
        # Linear with steps, memory overhead is minimal
        param_factor = max_steps / 100
        
    else:  # Hierarchical Beam Search
        # Beam Search: O(b^d)
        # Fixed complexity, independent of population parameters
        param_factor = 1.0
    
    # === FINAL CALCULATION ===
    estimated_time = baseline_time * network_factor * target_factor * evidence_factor * param_factor
    
    # Apply reasonable bounds
    return max(0.05, min(3600, estimated_time))  # Between 0.05s and 1 hour

def format_time_estimate(seconds):
    """Format time estimate for display"""
    if seconds < 1:
        return f"~{seconds:.1f}s"
    elif seconds < 60:
        return f"~{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"~{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"~{hours:.1f}h"

# Create session components - but use dynamic session creation
if SESSION_MANAGEMENT_AVAILABLE:
    # Don't create session here - will be created dynamically per user
    session_components = html.Div([
        # Dynamic session store - will be populated by callback
        dcc.Store(id='session-id-store', data=None),
        dcc.Store(id='heartbeat-counter', data=0),
        
        # Interval component for heartbeat (every 5 seconds)
        dcc.Interval(
            id='heartbeat-interval',
            interval=5*1000,  # 5 seconds
            n_intervals=0,
            disabled=False
        ),
        
        # Interval for cleanup check (every 30 seconds)
        dcc.Interval(
            id='cleanup-interval', 
            interval=30*1000,  # 30 seconds
            n_intervals=0,
            disabled=False
        ),
        
        # Hidden div for status
        html.Div(id='session-status', style={'display': 'none'}),
        
        # Client-side script for session management
        html.Script("""
            // Generate unique session ID per browser
            if (!window.dashSessionId) {
                window.dashSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }
            
            // Send heartbeat on page activity
            document.addEventListener('click', function() {
                if (window.dashHeartbeat) window.dashHeartbeat();
            });
            
            document.addEventListener('keypress', function() {
                if (window.dashHeartbeat) window.dashHeartbeat();
            });
            
            // Handle page unload
            window.addEventListener('beforeunload', function() {
                if (navigator.sendBeacon) {
                    navigator.sendBeacon('/dash/_disconnect', JSON.stringify({
                        session_id: window.dashSessionId
                    }));
                }
            });
            
            // Handle iframe unload (when parent page changes)
            if (window.parent !== window) {
                try {
                    window.parent.addEventListener('beforeunload', function() {
                        if (navigator.sendBeacon) {
                            navigator.sendBeacon('/dash/_disconnect', JSON.stringify({
                                session_id: window.dashSessionId
                            }));
                        }
                    });
                } catch(e) {
                    console.log('Cross-origin iframe detected');
                }
            }
        """),
    ], style={'display': 'none'})
    session_id = None  # Will be set dynamically
else:
    session_id = None
    session_components = html.Div()

# Add notification store and container
app.layout = html.Div([
    # Safari Compatibility Fix - inject CSS
    html.Div([
        dcc.Markdown(SAFARI_FIX_CSS, dangerously_allow_html=True)
    ], style={'display': 'none'}),
    
    # SESSION MANAGEMENT COMPONENTS - ADD THESE TO ALL DASH APPS
    session_components,
    
    dcc.Store(id='notification-store'),
    html.Div(id='notification-container', style={
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'zIndex': '1000',
        'width': '300px',
        'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(100%)',
        'opacity': '0'
    }),
    dcc.Loading(
        id="global-spinner",
        type="default",
        fullscreen=False,
        color="#00A2E1",
        style={
            "position": "fixed",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "zIndex": "999999"
        },
        children=html.Div([
            html.H1("Bayesian Network Optimization", style={'textAlign': 'center'}),

            html.Div(
                className="link-bar",
                style={"textAlign": "center", "marginBottom": "20px"},
                children=[
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Original GitHub"
                        ],
                        href="https://github.com/DanielZaragozaP/Most-Relevant-Explanation-in-BNs-with-EDAs",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Paper PDF"
                        ],
                        href="https://cig.fi.upm.es/wp-content/uploads/TFM_DANIEL_ZARAGOZA_PELLICER.pdf#chapter.4",
                        target="_blank",
                        className="btn btn-outline-primary me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Dash Adapted GitHub"
                        ],
                        href="https://github.com/KeevinPR/Most-Relevant-Explanation-in-BNs-with-EDAs",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                ]
            ),

            html.Div(
                [
                    html.P(
                        "The solution space of most relevant explanation (MRE) is very large and therefore to"
                        "find it is a very computationally expensive task as the number of nodes and states"
                        "of the BN grows",
                        style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto"}
                    )
                ],
                style={"marginBottom": "20px"}
            ),

            # (A) BIF Upload or use default network
            html.Div(className="card", children=[
                html.Div([
                    html.H3("1. Load Bayesian Network (.bif)", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-bif-upload",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                html.Div([
                    html.Div([
                        html.Img(
                            src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",
                            className="upload-icon"
                        ),
                        html.Div("Drag and drop or select a .bif file", className="upload-text")
                    ]),
                    dcc.Upload(
                        id='upload-bif',
                        children=html.Div([], style={'display': 'none'}),
                        className="upload-dropzone",
                        multiple=False
                    ),
                ], className="upload-card"),

                # The checkbox that is NOT checked by default:
                html.Div([
                    dcc.Checklist(
                        id='use-default-network',
                        options=[{'label': 'Use default Asia network (asia.bif)', 'value': 'default'}],
                        value=[],  # <-- Not checked by default
                        style={'display': 'inline-block', 'marginTop': '10px'}
                    ),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-default-dataset",
                        color="link",
                        style={"display": "inline-block", "marginLeft": "8px"}
                    ),
                html.Div(id='upload-status', style={'textAlign': 'center', 'color': 'green'}),
                ], style={'textAlign': 'center'}),
            ]),

            # Add BIF Upload Popover
            dbc.Popover(
                [
                    dbc.PopoverHeader(
                        [
                            "Bayesian Network Requirements",
                            html.I(className="fa fa-check-circle ms-2", style={"color": "#198754"})
                        ],
                        style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
                    ),
                    dbc.PopoverBody(
                        [
                            html.Ul([
                                html.Li(
                                    children=[
                                        html.Strong("Format: "),
                                        "BIF (Bayesian Interchange Format) file"
                                    ]
                                ),
                                html.Li(
                                    children=[
                                        html.Strong("Structure: "),
                                        "Must be a valid Bayesian network with nodes and conditional probability tables"
                                    ]
                                ),
                                html.Li(
                                    children=[
                                        html.Strong("Nodes: "),
                                        "Each node must have defined states and probabilities"
                                    ]
                                ),
                                html.Li(
                                    children=[
                                        html.Strong("Default: "),
                                        "You can use the default Asia network for testing"
                                    ]
                                ),
                            ]),
                        ],
                        style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "300px"}
                    ),
                ],
                id="help-popover-bif-upload",
                target="help-button-bif-upload",
                placement="right",
                is_open=False,
                trigger="hover",
                style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
            ),

            # (B) Evidence and Targets
            html.Div(className="card", children=[
                html.Div([
                    html.H3("2. Select Evidence", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-evidence",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                
                # Buttons for bulk selection
                html.Div([
                    dbc.Button(
                        "Select All",
                        id="select-all-evidence",
                        color="outline-primary",
                        size="sm",
                        style={'marginRight': '10px'}
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-evidence",
                        color="outline-secondary",
                        size="sm"
                    )
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                
                # Checkbox container for evidence variables
                html.Div(
                    id='evidence-checkbox-container',
                    style={
                        'maxHeight': '200px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'margin': '0 auto',
                        'width': '80%',
                        'backgroundColor': '#f8f9fa'
                    }
                ),
                
                html.Div(id='evidence-values-container')
            ]),

            # Add Evidence Selection Popover
            dbc.Popover(
                [
                    dbc.PopoverHeader(
                        [
                            "Evidence Selection",
                            html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                        ],
                        style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
                    ),
                    dbc.PopoverBody(
                        [
                            html.P("Evidence variables are the known states in your Bayesian network."),
                            html.P("Select one or more variables that you want to use as evidence."),
                            html.P("For each selected variable, you'll need to specify its state."),
                            html.P("These values will be used to compute the Most Relevant Explanation."),
                        ],
                        style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "300px"}
                    ),
                ],
                id="help-popover-evidence",
                target="help-button-evidence",
                placement="right",
                is_open=False,
                trigger="hover",
                style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
            ),

            html.Div(className="card", children=[
                html.Div([
                    html.H3("3. Select Target Variables", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-targets",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                
                # Buttons for bulk selection
                html.Div([
                    dbc.Button(
                        "Select All",
                        id="select-all-targets",
                        color="outline-primary",
                        size="sm",
                        style={'marginRight': '10px'}
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-targets",
                        color="outline-secondary",
                        size="sm"
                    )
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                
                # Checkbox container for target variables
                html.Div(
                    id='target-checkbox-container',
                    style={
                        'maxHeight': '200px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'margin': '0 auto',
                        'width': '80%',
                        'backgroundColor': '#f8f9fa'
                    }
                ),
                
                # Info message about intelligent selection
                html.Div([
                    html.I(className="fa fa-info-circle", style={'marginRight': '5px', 'color': '#6c757d'}),
                    html.Span("Target variables automatically exclude evidence variables. Previous selections are preserved when possible.", 
                             style={'fontSize': '11px', 'color': '#6c757d'})
                ], style={'textAlign': 'center', 'marginTop': '8px'}),
                
            ]),

            # Add Target Variables Popover
            dbc.Popover(
                [
                    dbc.PopoverHeader(
                        [
                            "Target Variables",
                            html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                        ],
                        style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
                    ),
                    dbc.PopoverBody(
                        [
                            html.P("Target variables are the nodes you want to explain in your Bayesian network."),
                            html.P("Select one or more variables that you want to find the Most Relevant Explanation for."),
                            html.P("Important notes:"),
                            html.Ul([
                                html.Li("Variables used as evidence cannot be selected as targets"),
                                html.Li("All algorithms can work with 1 or more target variables"),
                                html.Li("The more targets you select, the more complex the computation becomes")
                            ]),
                        ],
                        style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "300px"}
                    ),
                ],
                id="help-popover-targets",
                target="help-button-targets",
                placement="right",
                is_open=False,
                trigger="hover",
                style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
            ),

            # Add Algorithms Popover
            dbc.Popover(
                [
                    dbc.PopoverHeader(
                        [
                            "Available Optimization Algorithms",
                            html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                        ],
                        style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
                    ),
                    dbc.PopoverBody(
                        [
                            html.H6("Estimation of Distribution Algorithms (EDAs):", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            html.Ul([
                                html.Li([html.Strong("UMDAcat_mre2: "), "Univariate Marginal Distribution Algorithm for Categorical variables (improved)"]),
                                html.Li([html.Strong("EBNA MRE: "), "Estimation of Bayesian Network Algorithm"]),
                            ], style={"marginBottom": "10px"}),
                            
                            html.H6("Evolutionary Algorithms:", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            html.Ul([
                                html.Li([html.Strong("GA MRE: "), "Genetic Algorithm"]),
                                html.Li([html.Strong("ES MRE: "), "Evolution Strategy"]),
                                html.Li([html.Strong("DEA MRE: "), "Differential Evolution Algorithm"]),
                                html.Li([html.Strong("NSGA2 MRE: "), "Non-dominated Sorting Genetic Algorithm II"]),
                            ], style={"marginBottom": "10px"}),
                            
                            html.H6("Other Metaheuristics:", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            html.Ul([
                                html.Li([html.Strong("PSO MRE: "), "Particle Swarm Optimization"]),
                                html.Li([html.Strong("Tabu MRE: "), "Tabu Search"]),
                                html.Li([html.Strong("Hierarchical Beam Search: "), "Tree-based search algorithm"]),
                            ]),
                            
                            html.Hr(),
                            html.P("All algorithms can work with 1 or more target variables.", style={"fontSize": "12px", "color": "#666", "marginTop": "10px", "marginBottom": "0px"}),
                        ],
                        style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "400px"}
                    ),
                ],
                id="help-popover-algorithms",
                target="help-button-algorithms",
                placement="right",
                is_open=False,
                trigger="hover",
                style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
            ),

            # (C) Algorithm parameters
            html.Div(className="card", children=[
                html.Div([
                    html.H3("4. Algorithm Parameters", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-parameters",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                html.Div([
                    html.Label('Population Size:', style={'marginRight': '10px'}),
                    dcc.Input(id='pop-size-input', type='number', value=50, min=1, step=1, style={'width': '60px'}),
                    html.Label('Number of Generations:', style={'marginLeft': '20px', 'marginRight': '10px'}),
                    dcc.Input(id='num-gen-input', type='number', value=50, min=1, step=1, style={'width': '60px'}),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Max Steps:', style={'marginRight': '10px'}),
                    dcc.Input(id='max-steps-input', type='number', value=100, min=1, step=1, style={'width': '60px'}),
                    html.Label('Dead Iterations:', style={'marginLeft': '20px', 'marginRight': '10px'}),
                    dcc.Input(id='dead-iter-input', type='number', value=10, min=1, step=1, style={'width': '60px'}),
                ], style={'textAlign': 'center'}),
            ]),

            # Add Algorithm Parameters Popover
            dbc.Popover(
                [
                    dbc.PopoverHeader(
                        [
                            "Algorithm Parameters",
                            html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                        ],
                        style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
                    ),
                    dbc.PopoverBody(
                        [
                            html.P("These parameters control the behavior of the optimization algorithms:"),
                            html.Ul([
                                html.Li([
                                    html.Strong("Population Size: "),
                                    "Number of solutions evaluated in each generation"
                                ]),
                                html.Li([
                                    html.Strong("Number of Generations: "),
                                    "Maximum number of iterations for the algorithm"
                                ]),
                                html.Li([
                                    html.Strong("Max Steps: "),
                                    "Maximum number of steps for algorithms like Tabu Search"
                                ]),
                                html.Li([
                                    html.Strong("Dead Iterations: "),
                                    "Number of iterations without improvement before stopping"
                                ])
                            ]),
                            html.P("Higher values may improve results but increase computation time."),
                        ],
                        style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "300px"}
                    ),
                ],
                id="help-popover-parameters",
                target="help-button-parameters",
                placement="right",
                is_open=False,
                trigger="hover",
                style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
            ),

            # (E) Algorithm Selection
            html.Div(className="card", children=[
                html.Div([
                    html.H3("5. Select Algorithms", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-algorithms",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                
                # Buttons for bulk selection
                html.Div([
                    dbc.Button(
                        "Select All",
                        id="select-all-algorithms",
                        color="outline-primary",
                        size="sm",
                        style={'marginRight': '10px'}
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-algorithms",
                        color="outline-secondary",
                        size="sm"
                    )
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                
                # Checkbox container for algorithms
                html.Div(
                    id='algorithm-checkbox-container',
                    style={
                        'maxHeight': '200px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'margin': '0 auto',
                        'width': '90%',
                        'backgroundColor': '#f8f9fa'
                    }
                ),
                
                # Info message about algorithm selection
                html.Div([
                    html.I(className="fa fa-info-circle", style={'marginRight': '5px', 'color': '#6c757d'}),
                    html.Span("Select one or more algorithms to run. Time estimates update automatically based on your selections and parameters.", 
                             style={'fontSize': '11px', 'color': '#6c757d'})
                ], style={'textAlign': 'center', 'marginTop': '8px'}),
                
                # Total time estimate display
                html.Div(id='total-time-estimate', style={'textAlign': 'center', 'marginTop': '10px'}),
            ]),

            # (F) "Run" button + progress messages
            html.Div([
                html.Div([
                    dbc.Button(
                        [
                            html.I(className="fas fa-play-circle me-2"),
                            "Run Optimization"
                        ],
                        id='run-optimization-button',
                        n_clicks=0,
                        color="info",
                        className="btn-lg",
                        style={
                            'fontSize': '1.1rem',
                            'padding': '0.75rem 2rem',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'transition': 'all 0.3s ease',
                            'backgroundColor': '#00A2E1',
                            'border': 'none',
                            'margin': '1rem 0',
                            'color': 'white',
                            'fontWeight': '500'
                        }
                    )
                ], style={'textAlign': 'center'}),
                dcc.Store(id='run-button-store')
            ], id='run-button-container', style={'textAlign': 'center'}),

            html.Br(),
            html.Div(id='optimization-results'),

            # (G) Interval for progress tracking, plus hidden stores
            html.Div(id='scroll-helper', style={'display': 'none'}),
            dcc.Store(id='stored-network'),
            dcc.Store(id='uploaded-bif-content'),
            dcc.Store(id='previous-evidence-selection', data=[]),
            dcc.Store(id='previous-target-selection', data=[]),
            dcc.Store(id='algorithm-selection-store', data=[]),  # Store for algorithm selections
            dcc.Interval(
                id='progress-interval',
                interval=1000,
                n_intervals=0,
                disabled=True
            )
        ])
    ),
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Help",
                    html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                ],
                style={
                    "backgroundColor": "#f8f9fa",  # Light gray background
                    "fontWeight": "bold"
                }
            ),
            dbc.PopoverBody(
                [
                    html.P(
                        [
                            "For details and content of the dataset, check out: ",
                            html.A(
                                "asia.bif",
                                href="https://github.com/KeevinPR/Most-Relevant-Explanation-in-BNs-with-EDAs/blob/main/asia.bif",
                                target="_blank",
                                style={"textDecoration": "underline", "color": "#0d6efd"}
                            ),
                        ]
                    ),
                    html.Hr(),  # Horizontal rule for a modern divider
                    html.P("Feel free to upload your own dataset at any time.")
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "0 0 0.25rem 0.25rem"
                }
            ),
        ],
        id="help-popover-default-dataset",
        target="help-button-default-dataset",
        placement="right",
        is_open=False,
        trigger="hover"
    ),
])

# Client-side callback for scrolling after results appear
app.clientside_callback(
    """
    function(results) {
        if (results) {
            setTimeout(function() {
                var elem = document.getElementById('optimization-results');
                if (elem) {
                    elem.scrollIntoView({behavior: 'smooth'});
                }
            }, 100);
        }
        return '';
    }
    """,
    Output('scroll-helper', 'children'),
    [Input('optimization-results', 'children')]
)

# Global model cache
cached_model = None

# Callback to handle .bif upload or default checkbox
@app.callback(
    Output('stored-network', 'data'),
    Output('upload-status', 'children'),
    Output('use-default-network', 'value'),
    Input('upload-bif', 'contents'),
    State('upload-bif', 'filename'),
    Input('use-default-network', 'value')
)
def load_network(contents, filename, use_default_value):
    """
    1) If user uploads a .bif, parse that and uncheck default checkbox.
    2) If user checks 'default' (and no file uploaded), load asia.bif from disk.
    3) Otherwise, no network is used (None).
    """
    global cached_model
    cached_model = None
    
    # PRIORITY 1: Handle file upload first
    if contents is not None:
        # The user uploaded a file - this takes priority
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            bif_data = decoded.decode('utf-8')
            # Validate by attempting to parse
            test_reader = BIFReader(string=bif_data)
            test_model = test_reader.get_model()  # just to confirm it's valid
            msg = f"Successfully loaded network from {filename}."
            logger.info(msg)
            return (
                {
                    'network_name': filename,
                    'network_type': 'string',
                    'content': bif_data
                },
                msg,
                []  # Uncheck the default checkbox when file is uploaded
            )
        except Exception as e:
            logger.error(f"Error loading network from {filename}: {e}")
            return None, f"Error loading {filename}: {e}", use_default_value
    
    # PRIORITY 2: Handle default checkbox (only if no file was uploaded)
    if 'default' in use_default_value:
        try:
            with open('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/MREWithEDAs/asia.bif', 'r') as f:
                default_data = f.read()
            msg = "Using default network: asia.bif"
            logger.info(msg)
            return (
                {
                    'network_name': 'asia.bif',
                    'network_type': 'string',
                    'content': default_data
                },
                msg,
                use_default_value  # Keep checkbox as is
            )
        except Exception as e:
            logger.error(f"Error reading default network: {e}")
            return None, f"Error reading default network: {e}", use_default_value

    # If neither default nor file upload
    return None, "No network selected. Upload a file or check the default option.", use_default_value

def get_model(stored_network):
    """
    Use the cached model if available. Otherwise parse from stored_network,
    or return None if stored_network is empty.
    """
    global cached_model
    if cached_model is not None:
        return cached_model
    if not stored_network:
        return None

    try:
        if stored_network['network_type'] == 'string':
            r = BIFReader(string=stored_network['content'])
            model = r.get_model()
            cached_model = model
            return model
        else:
            logger.error("Invalid network_type in stored_network.")
            return None
    except Exception as e:
        logger.error(f"Could not parse the network: {e}")
        return None

# Populate the evidence dropdown only if a model is available
@app.callback(
    Output('evidence-checkbox-container', 'children'),
    Input('stored-network', 'data')
)
def update_evidence_variables(stored_network):
    m = get_model(stored_network)
    if not m:
        return html.Div("No network loaded", style={'textAlign': 'center', 'color': '#666'})
    
    variables = list(m.nodes())
    if not variables:
        return html.Div("No variables found", style={'textAlign': 'center', 'color': '#666'})
    
    # Create checkboxes in a grid layout
    checkboxes = []
    for i, var in enumerate(variables):
        checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'evidence-checkbox', 'index': var},
                    options=[{'label': f' {var}', 'value': var}],
                    value=[],
                    style={'margin': '0'}
                )
            ], style={'display': 'inline-block', 'width': '50%', 'marginBottom': '5px'})
        )
    
    return html.Div(checkboxes, style={'columnCount': '2', 'columnGap': '20px'})

# Build the dynamic evidence-value dropdowns
@app.callback(
    Output('evidence-values-container', 'children'),
    Input({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
    State('stored-network', 'data')
)
def update_evidence_values(checkbox_values, stored_network):
    # Get selected evidence variables from checkboxes
    ctx = dash.callback_context
    if not ctx.inputs:
        return []
    
    # Extract selected variables
    evidence_vars = []
    for input_info in ctx.inputs_list[0]:
        if input_info['value']:  # If checkbox is checked
            var_name = input_info['id']['index']
            evidence_vars.append(var_name)
    
    if not evidence_vars:
        return []

    m = get_model(stored_network)
    if m is None:
        return []

    children = []
    for var in evidence_vars:
        states = m.get_cpds(var).state_names[var]
        children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                f"Select value for {var}",
                                style={'width': '40%', 'textAlign': 'right', 'paddingRight': '10px'}
                            ),
                            dcc.Dropdown(
                                id={'type': 'evidence-value-dropdown', 'index': var},
                                options=[{'label': s, 'value': s} for s in states],
                                value=states[0] if states else None,
                                style={'width': '60%'}
                            )
                        ],
                        style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
                    )
                ],
                style={'marginBottom': '10px', 'width': '50%', 'margin': '0 auto'}
            )
        )
    return children

# Populate target variables, excluding those in evidence, ignoring single-state
@app.callback(
    Output('target-checkbox-container', 'children'),
    Output('previous-evidence-selection', 'data'),
    Input({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
    State('stored-network', 'data'),
    State('previous-evidence-selection', 'data'),
    State('previous-target-selection', 'data')
)
def update_target_options(checkbox_values, stored_network, prev_evidence, prev_targets):
    m = get_model(stored_network)
    if m is None:
        return html.Div("No network loaded", style={'textAlign': 'center', 'color': '#666'}), []
    
    # Get currently selected evidence variables from checkboxes
    current_evidence = []
    ctx = dash.callback_context
    if ctx.inputs_list and ctx.inputs_list[0]:
        for input_info in ctx.inputs_list[0]:
            if input_info['value']:  # If checkbox is checked
                var_name = input_info['id']['index']
                current_evidence.append(var_name)
    
    all_vars = set(m.nodes())
    available = [v for v in all_vars if v not in current_evidence]

    valid_targets = []
    for v in available:
        states = m.get_cpds(v).state_names[v]
        if len(states) > 1:
            valid_targets.append(v)

    if not valid_targets:
        return html.Div("No target variables available", style={'textAlign': 'center', 'color': '#666'}), current_evidence
    
    # Calculate which targets should remain selected:
    # 1. Variables that were targets before and are still available
    # 2. Variables that were removed from evidence and were targets before
    newly_available = set(prev_evidence) - set(current_evidence)  # Variables removed from evidence
    keep_selected = (set(prev_targets) & set(valid_targets)) | (newly_available & set(prev_targets))
    
    # Create checkboxes in a grid layout
    checkboxes = []
    for var in valid_targets:
        # Pre-select if it should remain selected
        initial_value = [var] if var in keep_selected else []
        
        checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'target-checkbox', 'index': var},
                    options=[{'label': f' {var}', 'value': var}],
                    value=initial_value,
                    style={'margin': '0'}
                )
            ], style={'display': 'inline-block', 'width': '50%', 'marginBottom': '5px'})
        )
    
    return html.Div(checkboxes, style={'columnCount': '2', 'columnGap': '20px'}), current_evidence

# Callback to track target selections for intelligent management
@app.callback(
    Output('previous-target-selection', 'data'),
    Input({'type': 'target-checkbox', 'index': ALL}, 'value')
)
def track_target_selections(target_checkbox_values):
    """Track which targets are currently selected"""
    selected_targets = []
    for checkbox_value in target_checkbox_values or []:
        if checkbox_value:  # If checkbox is checked
            selected_targets.extend(checkbox_value)
    return selected_targets

# Callback to run optimization
@app.callback(
    Output('optimization-results', 'children'),
    Output('progress-interval', 'disabled'),
    Output('run-button-store', 'data'),
    Input('run-optimization-button', 'n_clicks'),
    State('stored-network', 'data'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'id'),
    State({'type': 'target-checkbox', 'index': ALL}, 'value'),
    State({'type': 'algorithm-checkbox', 'index': ALL}, 'value'),
    State('pop-size-input', 'value'),
    State('num-gen-input', 'value'),
    State('max-steps-input', 'value'),
    State('dead-iter-input', 'value')
)
def run_optimization(n_clicks,
                     stored_network,
                     evidence_values, evidence_ids,
                     target_checkbox_values,
                     algorithm_checkbox_values,
                     pop_size, n_gen, max_steps, dead_iter):
    if not n_clicks:
        raise PreventUpdate

    # REGISTER THIS PROCESS WITH SESSION MANAGER (CRITICAL FOR CLEANUP)
    if SESSION_MANAGEMENT_AVAILABLE and session_id:
        register_long_running_process(session_id)
        logger.info(f"Registered optimization process for session {session_id}")

    model = get_model(stored_network)
    if model is None:
        return html.Div("No network is currently selected. Please upload or choose the default option.",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # We'll enable the progress interval for real-time messages
    interval_disabled = False
    hide_run_button = True

    # Build evidence dictionary
    evidence_dict = {}
    for ident, val in zip(evidence_ids, evidence_values):
        evidence_dict[ident['index']] = val

    # Get selected target variables from checkboxes
    target_vars = []
    for checkbox_value in target_checkbox_values:
        if checkbox_value:  # If checkbox is checked, it contains the variable name
            target_vars.extend(checkbox_value)

    # Get selected algorithms from checkboxes
    selected_algorithms = []
    for checkbox_value in algorithm_checkbox_values:
        if checkbox_value:  # If checkbox is checked, it contains the algorithm name
            selected_algorithms.extend(checkbox_value)

    # Check if at least one algorithm is selected
    if not selected_algorithms:
        return html.Div("No algorithms selected. Please select at least one algorithm to run.",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # Check for overlap between evidence and target
    overlap = set(evidence_dict.keys()) & set(target_vars or [])
    if overlap:
        return html.Div(f"Variables cannot be both evidence and target: {', '.join(overlap)}",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # Filter out any target variable that is deterministic given the evidence
    inference = VariableElimination(model)
    final_targets = []
    eps = 1e-8
    for t in (target_vars or []):
        try:
            q = inference.query([t], evidence=evidence_dict, show_progress=False)
            dist = q.values.flatten()
            if max(dist) >= 1 - eps:
                update_progress(f"Excluding {t} from targets (deterministic given evidence).")
            else:
                final_targets.append(t)
        except Exception as exc:
            update_progress(f"Failed to query {t}: {exc}")

    if not final_targets:
        return html.Div("No valid target variables (all deterministic or none selected).",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # Run the selected algorithms (this is the long-running part)
    results = run_all_algorithms(model, evidence_dict, final_targets, pop_size, n_gen, max_steps, dead_iter, selected_algorithms)

    # Once finished, disable interval and show the run button again
    interval_disabled = True
    hide_run_button = False

   # 1) Build a DataFrame from your results list
    df = pd.DataFrame(results)

    # Rename the time column for clarity
    if 'Time' in df.columns:
        df = df.rename(columns={'Time': 'Time (s)'})

    # Sort by GBF (descending - higher is better) first, then by Time (ascending - faster is better) as tiebreaker
    if 'GBF' in df.columns and 'Time (s)' in df.columns and not df.empty:
        # Convert GBF to numeric for proper sorting (handle any potential string values)
        df['GBF_numeric'] = pd.to_numeric(df['GBF'], errors='coerce')
        df = df.sort_values(['GBF_numeric', 'Time (s)'], ascending=[False, True]).reset_index(drop=True)
        df = df.drop('GBF_numeric', axis=1)  # Remove the temporary numeric column

    # Serialize dicts and format numbers
    df['Solution'] = df['Solution'].apply(lambda v: json.dumps(v) if isinstance(v, dict) else str(v))
    df['GBF']      = df['GBF'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
    df['Time (s)'] = df['Time (s)'].apply(lambda t: f"{t:.2f}" if isinstance(t, (int, float)) else str(t))

    # 2) Create a responsive Bootstrap table
    table = dbc.Table.from_dataframe(
        df[['Algorithm', 'Solution', 'GBF', 'Time (s)']],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        className="mt-2"
    )

    # 3) Wrap it in your existing "card-big" class with centered title
    card = dbc.Card(
            dbc.CardBody([
                html.H4("Algorithm Results", className="card-title", style={'textAlign': 'center'}),
                table
            ])
        )
    return card, interval_disabled, hide_run_button

@app.callback(
    Output('run-optimization-button', 'style'),
    Input('run-button-store', 'data')
)
def hide_run_button_callback(flag):
    if flag:
        return {'display': 'none'}
    return {'display': 'inline-block'}

def update_progress(msg):
    # Solo mantenemos esto para logs del backend
    logger.info(f"Progress: {msg}")

def run_all_algorithms(model, evidence, targets, pop_size, n_gen, max_steps, dead_iter, selected_algorithms):
    logger.info("Starting run_all_algorithms")
    update_progress("Starting run_all_algorithms")

    results = []

    # Build a list of algorithms that have enough target variables AND are selected by user
    algorithms = []
    for a, min_t in algorithm_requirements.items():
        if len(targets) >= min_t and a in selected_algorithms:
            algorithms.append(a)
        elif len(targets) < min_t:
            update_progress(f"Skipping {a}; requires >= {min_t} targets.")
        # Note: we don't log anything for unselected algorithms to avoid clutter

    if not targets:
        update_progress("No targets found to optimize.")
        return results

    if not algorithms:
        update_progress("No valid algorithms selected for the given targets.")
        return results

    for alg in algorithms:
        try:
            if alg == 'UMDAcat_mre2':
                update_progress("Running UMDAcat_mre2")
                start = time.time()
                sol, gbf, _ = mre.UMDAcat_mre2(
                    model,
                    evidence,
                    targets,
                    size_gen=pop_size,
                    dead_iter=dead_iter,
                    verbose=False,
                    alpha=0.8,
                    best_init=True,
                    more_targets=0
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("UMDAcat_mre2 completed successfully.")

            elif alg == 'DEA MRE':
                update_progress("Running DEA MRE")
                start = time.time()
                sol, gbf = mre.dea_mre(model, evidence, targets, pop_size, max_steps, more_targets=0)
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("DEA MRE completed successfully.")

            elif alg == 'EBNA MRE':
                update_progress("Running EBNA MRE")
                start = time.time()
                sol, gbf, _ = mre.ebna_mre(
                    model,
                    evidence,
                    targets,
                    size_gen=pop_size,
                    dead_iter=dead_iter,
                    verbose=False,
                    alpha=0.8,
                    best_init=True,
                    more_targets=0
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("EBNA MRE completed successfully.")

            elif alg == 'ES MRE':
                update_progress("Running ES MRE")
                start = time.time()
                sol, gbf = mre.es_mre(model, evidence, targets, pop_size, max_steps, more_targets=0)
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("ES MRE completed successfully.")

            elif alg == 'GA MRE':
                update_progress("Running GA MRE")
                start = time.time()
                sol, gbf = mre.ga_mre(model, evidence, targets, pop_size, max_steps, more_targets=0)
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("GA MRE completed successfully.")

            elif alg == 'Hierarchical Beam Search':
                update_progress("Running Hierarchical Beam Search")
                start = time.time()
                sol, gbf = mre.hierarchical_beam_search(
                    model,
                    evidence,
                    targets,
                    5,           # beam_width
                    1 + 1e-8,    # delta
                    10,          # max_steps
                    2,
                    more_targets=0
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("Hierarchical Beam Search completed successfully.")

            elif alg == 'NSGA2 MRE':
                update_progress("Running NSGA2 MRE")
                start = time.time()
                sol, gbf = mre.nsga2_mre(
                    model, evidence, targets,
                    pop_size=pop_size, n_gen=n_gen,
                    best_init=True, period=10, more_targets=0
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("NSGA2 MRE completed successfully.")

            elif alg == 'PSO MRE':
                update_progress("Running PSO MRE")
                start = time.time()
                sol, gbf = mre.pso_mre(model, evidence, targets, pop_size, max_steps, more_targets=0)
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("PSO MRE completed successfully.")

            elif alg == 'Tabu MRE':
                update_progress("Running Tabu MRE")
                start = time.time()
                sol, gbf = mre.tabu_mre(
                    model, evidence, targets,
                    max_steps=max_steps,
                    tabu_size=30,
                    more_targets=0
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("Tabu MRE completed successfully.")

        except Exception as ex:
            msg = f"{alg} failed: {ex}"
            update_progress(msg)
            results.append({'Algorithm': alg, 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    update_progress("Finished run_all_algorithms")
    return results

# Add callback for BIF upload popover
@app.callback(
    Output("help-popover-bif-upload", "is_open"),
    Input("help-button-bif-upload", "n_clicks"),
    State("help-popover-bif-upload", "is_open")
)
def toggle_bif_upload_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# Add callback for evidence popover
@app.callback(
    Output("help-popover-evidence", "is_open"),
    Input("help-button-evidence", "n_clicks"),
    State("help-popover-evidence", "is_open")
)
def toggle_evidence_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# Add callback for target variables popover
@app.callback(
    Output("help-popover-targets", "is_open"),
    Input("help-button-targets", "n_clicks"),
    State("help-popover-targets", "is_open")
)
def toggle_targets_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# Add callback for algorithms popover
@app.callback(
    Output("help-popover-algorithms", "is_open"),
    Input("help-button-algorithms", "n_clicks"),
    State("help-popover-algorithms", "is_open")
)
def toggle_algorithms_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# Add callback for algorithm parameters popover
@app.callback(
    Output("help-popover-parameters", "is_open"),
    Input("help-button-parameters", "n_clicks"),
    State("help-popover-parameters", "is_open")
)
def toggle_parameters_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# Callbacks for evidence selection buttons
@app.callback(
    Output({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
    [Input('select-all-evidence', 'n_clicks'),
     Input('clear-evidence', 'n_clicks')],
    [State({'type': 'evidence-checkbox', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_evidence_selection(select_all_clicks, clear_clicks, checkbox_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'select-all-evidence':
        # Select all checkboxes
        return [[checkbox_id['index']] for checkbox_id in checkbox_ids]
    elif button_id == 'clear-evidence':
        # Clear all checkboxes
        return [[] for _ in checkbox_ids]
    
    raise PreventUpdate

# Callbacks for target selection buttons
@app.callback(
    Output({'type': 'target-checkbox', 'index': ALL}, 'value'),
    [Input('select-all-targets', 'n_clicks'),
     Input('clear-targets', 'n_clicks')],
    [State({'type': 'target-checkbox', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_target_selection(select_all_clicks, clear_clicks, checkbox_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'select-all-targets':
        # Select all checkboxes
        return [[checkbox_id['index']] for checkbox_id in checkbox_ids]
    elif button_id == 'clear-targets':
        # Clear all checkboxes
        return [[] for _ in checkbox_ids]
    
    raise PreventUpdate

# Callbacks for algorithm selection buttons
@app.callback(
    Output({'type': 'algorithm-checkbox', 'index': ALL}, 'value'),
    [Input('select-all-algorithms', 'n_clicks'),
     Input('clear-algorithms', 'n_clicks')],
    [State({'type': 'algorithm-checkbox', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_algorithm_selection(select_all_clicks, clear_clicks, checkbox_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'select-all-algorithms':
        # Select all checkboxes
        return [[checkbox_id['index']] for checkbox_id in checkbox_ids]
    elif button_id == 'clear-algorithms':
        # Clear all checkboxes
        return [[] for _ in checkbox_ids]
    
    raise PreventUpdate

# Add notification callback
@app.callback(
    [Output('notification-container', 'children'),
     Output('notification-container', 'style')],
    Input('notification-store', 'data')
)
def show_notification(data):
    if data is None:
        return None, {
            'position': 'fixed',
            'bottom': '20px',
            'right': '20px',
            'zIndex': '1000',
            'width': '300px',
            'transition': 'all 0.3s ease-in-out',
            'transform': 'translateY(100%)',
            'opacity': '0'
        }
    
    # Create toast with animation
    toast = dbc.Toast(
        data['message'],
        header=data['header'],
        icon=data['icon'],
        is_open=True,
        dismissable=True,
        style={
            'width': '100%',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'borderRadius': '8px',
            'marginBottom': '10px'
        }
    )
    
    # Style to show notification with animation
    container_style = {
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'zIndex': '1000',
        'width': '300px',
        'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(0)',
        'opacity': '1'
    }
    
    return toast, container_style

def show_error(message, header="Error"):
    """Show error notification"""
    return {
        'message': message,
        'header': header,
        'icon': 'danger'
    }

def show_success(message, header="Success"):
    """Show success notification"""
    return {
        'message': message,
        'header': header,
        'icon': 'success'
    }

def show_warning(message, header="Warning"):
    """Show warning notification"""
    return {
        'message': message,
        'header': header,
        'icon': 'warning'
    }

def show_info(message, header="Information"):
    return {
        'children': [
            dbc.Alert([
                html.I(className="fa fa-info-circle me-2"),
                message
            ], color="info", dismissable=True)
        ],
        'style': {
            'position': 'fixed',
            'bottom': '20px',
            'right': '20px',
            'zIndex': '1000',
            'width': '300px',
            'opacity': '1',
            'transform': 'translateY(0%)'
        }
    }

# Callback to track algorithm selections for persistence
@app.callback(
    Output('algorithm-selection-store', 'data'),
    Input({'type': 'algorithm-checkbox', 'index': ALL}, 'value')
)
def track_algorithm_selections(algorithm_checkbox_values):
    """Track which algorithms are currently selected"""
    selected_algorithms = []
    for checkbox_value in algorithm_checkbox_values or []:
        if checkbox_value:
            selected_algorithms.extend(checkbox_value)
    return selected_algorithms

# Populate algorithm checkboxes - all available by default
@app.callback(
    Output('algorithm-checkbox-container', 'children'),
    [Input('stored-network', 'data'),
     Input({'type': 'target-checkbox', 'index': ALL}, 'value'),
     Input({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
     Input('pop-size-input', 'value'),
     Input('num-gen-input', 'value'),
     Input('max-steps-input', 'value'),
     Input('dead-iter-input', 'value')],
    State('algorithm-selection-store', 'data')
)
def update_algorithm_checkboxes(stored_network, target_checkbox_values, evidence_checkbox_values, 
                               pop_size, n_gen, max_steps, dead_iter, current_selections):
    """Create checkboxes for all available algorithms with time estimates"""
    available_algorithms = [
        'UMDAcat_mre2',
        'EBNA MRE', 
        'DEA MRE',
        'ES MRE',
        'GA MRE',
        'NSGA2 MRE',
        'PSO MRE',
        'Tabu MRE',
        'Hierarchical Beam Search'
    ]
    
    # Get network information
    model = get_model(stored_network)
    num_nodes = len(model.nodes()) if model else 8  # Default to 8 if no model
    
    # Count selected targets
    num_targets = 0
    for checkbox_value in (target_checkbox_values or []):
        if checkbox_value:
            num_targets += len(checkbox_value)
    num_targets = max(1, num_targets)  # At least 1 target for estimation
    
    # Count selected evidence
    num_evidence = 0
    for checkbox_value in (evidence_checkbox_values or []):
        if checkbox_value:
            num_evidence += len(checkbox_value)
    
    # Default values if None
    pop_size = pop_size or 50
    n_gen = n_gen or 50
    max_steps = max_steps or 100
    dead_iter = dead_iter or 10
    
    # Use current selections if available, otherwise default to all selected
    if current_selections is None:
        current_selections = available_algorithms  # All selected by default
    
    # Create checkboxes in a 2-column layout with time estimates
    checkboxes = []
    for i, algorithm in enumerate(available_algorithms):
        # Calculate time estimate
        time_estimate = estimate_algorithm_time(
            algorithm, num_targets, num_evidence, num_nodes, 
            pop_size, n_gen, max_steps, dead_iter
        )
        formatted_time = format_time_estimate(time_estimate)
        
        # Determine if this algorithm should be selected
        is_selected = algorithm in current_selections
        
        # Create checkbox with time estimate
        checkbox_div = html.Div([
            html.Div([
                dcc.Checklist(
                    id={'type': 'algorithm-checkbox', 'index': algorithm},
                    options=[{'label': f' {algorithm}', 'value': algorithm}],
                    value=[algorithm] if is_selected else [],
                    style={'margin': '0', 'display': 'inline-block'}
                ),
                html.Span(
                    formatted_time,
                    style={
                        'marginLeft': '10px',
                        'fontSize': '11px',
                        'color': '#6c757d',
                        'fontStyle': 'italic',
                        'float': 'right'
                    }
                )
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'width': '100%',
                'padding': '2px 5px',
                'borderRadius': '3px',
                'backgroundColor': '#f8f9fa'
            })
        ], style={
            'width': '48%', 
            'marginBottom': '5px', 
            'marginRight': '2%' if i % 2 == 0 else '0',
            'display': 'inline-block',
            'verticalAlign': 'top'
        })
        
        checkboxes.append(checkbox_div)
    
    return html.Div(checkboxes, style={'width': '100%'})

# Callback to show total estimated time
@app.callback(
    Output('total-time-estimate', 'children'),
    [Input({'type': 'algorithm-checkbox', 'index': ALL}, 'value'),
     Input('stored-network', 'data'),
     Input({'type': 'target-checkbox', 'index': ALL}, 'value'),
     Input({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
     Input('pop-size-input', 'value'),
     Input('num-gen-input', 'value'),
     Input('max-steps-input', 'value'),
     Input('dead-iter-input', 'value')]
)
def update_total_time_estimate(algorithm_checkbox_values, stored_network, target_checkbox_values, 
                              evidence_checkbox_values, pop_size, n_gen, max_steps, dead_iter):
    """Calculate and display total estimated time for selected algorithms"""
    
    # Get selected algorithms
    selected_algorithms = []
    for checkbox_value in algorithm_checkbox_values or []:
        if checkbox_value:
            selected_algorithms.extend(checkbox_value)
    
    if not selected_algorithms:
        return html.Div()
    
    # Get network information
    model = get_model(stored_network)
    num_nodes = len(model.nodes()) if model else 8
    
    # Count selected targets and evidence
    num_targets = 0
    for checkbox_value in (target_checkbox_values or []):
        if checkbox_value:
            num_targets += len(checkbox_value)
    num_targets = max(1, num_targets)
    
    num_evidence = 0
    for checkbox_value in (evidence_checkbox_values or []):
        if checkbox_value:
            num_evidence += len(checkbox_value)
    
    # Default values if None
    pop_size = pop_size or 50
    n_gen = n_gen or 50
    max_steps = max_steps or 100
    dead_iter = dead_iter or 10
    
    # Calculate total time
    total_time = 0
    for algorithm in selected_algorithms:
        time_estimate = estimate_algorithm_time(
            algorithm, num_targets, num_evidence, num_nodes, 
            pop_size, n_gen, max_steps, dead_iter
        )
        total_time += time_estimate
    
    formatted_total_time = format_time_estimate(total_time)
    
    return html.Div([
        html.I(className="fa fa-clock-o", style={'marginRight': '5px', 'color': '#17a2b8'}),
        html.Strong(f"Total estimated time: {formatted_total_time}", 
                   style={'color': '#17a2b8', 'fontSize': '13px'})
    ], style={
        'backgroundColor': '#d4edda',
        'border': '1px solid #c3e6cb',
        'borderRadius': '5px',
        'padding': '8px',
        'marginTop': '10px',
        'display': 'inline-block'
    })

# Setup session management callbacks
if SESSION_MANAGEMENT_AVAILABLE:
    @app.callback(
        Output('session-id-store', 'data'),
        Input('heartbeat-interval', 'n_intervals'),
        State('session-id-store', 'data'),
        prevent_initial_call=False
    )
    def initialize_session(n_intervals, stored_session_id):
        """Initialize session ID dynamically for each user."""
        if stored_session_id is None:
            # Create new session for this user
            session_manager = get_session_manager()
            new_session_id = session_manager.register_session()
            session_manager.register_process(new_session_id, os.getpid())
            logger.info(f"New MRE session created: {new_session_id}")
            return new_session_id
        return stored_session_id
    
    @app.callback(
        Output('session-status', 'children'),
        Input('heartbeat-interval', 'n_intervals'),
        State('session-id-store', 'data'),
        prevent_initial_call=True
    )
    def send_heartbeat(n_intervals, session_id):
        """Send heartbeat to session manager."""
        if session_id:
            session_manager = get_session_manager()
            session_manager.heartbeat(session_id)
            if n_intervals % 12 == 0:  # Log every minute (every 12 intervals of 5s)
                logger.info(f"MRE heartbeat sent for session: {session_id}")
            return f"Heartbeat sent: {n_intervals}"
        return "No session"
    
    @app.callback(
        Output('heartbeat-counter', 'data'),
        Input('cleanup-interval', 'n_intervals'),
        State('session-id-store', 'data'),
        prevent_initial_call=True
    )
    def periodic_cleanup_check(n_intervals, session_id):
        """Periodic check to ensure session is still active."""
        if session_id:
            session_manager = get_session_manager()
            active_sessions = session_manager.get_active_sessions()
            if session_id not in active_sessions:
                # Session expired, try to refresh or handle gracefully
                logger.warning(f"MRE session {session_id} expired")
                return n_intervals
        return n_intervals

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8052)