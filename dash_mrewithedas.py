import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import time
import json
import random
import warnings
import logging
import base64
import threading
from dash.exceptions import PreventUpdate

from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination

# Import your custom module containing the optimization algorithms
import mre  # Ensure this is accessible and correctly installed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    requests_pathname_prefix='/Reasoning/MREWithEDAsDash/',
    suppress_callback_exceptions=True
)

progress_lock = threading.Lock()
progress_messages = []

# Load the default Bayesian Network model
reader = BIFReader('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/MREWithEDAs/asia.bif')  # Ensure 'asia.bif' is in the same directory
default_model = reader.get_model()

app.layout = html.Div([
    html.H1("Bayesian Network Optimization", style={'textAlign': 'center'}),
    html.Hr(),
    
    # File upload component
    html.Div([
        html.H3("Upload a .bif File or Use the Default Network", style={'textAlign': 'center'}),
        dcc.Upload(
            id='upload-bif',
            children=html.Button('Upload .bif File'),
            style={'textAlign': 'center'}
        ),
        html.Br(),
        dcc.Checklist(
            id='use-default-network',
            options=[{'label': ' Use Default Network (asia.bif)', 'value': 'default'}],
            value=['default'],  # Default to using the asia.bif
            style={'textAlign': 'center'}
        ),
    ], style={'textAlign': 'center'}),
    
    html.Hr(),
    html.Div([
        html.H3("Select Evidence Variables", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='evidence-vars-dropdown',
            options=[{'label': var, 'value': var} for var in default_model.nodes()],
            multi=True,
            placeholder="Select evidence variables",
            style={'width': '50%', 'margin': '0 auto'}
        ),
        html.Div(id='evidence-values-container')
    ], style={'marginBottom': '20px'}),
    html.Hr(),
    html.Div([
        html.H3("Select Target Variables", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='target-vars-dropdown',
            options=[{'label': var, 'value': var} for var in default_model.nodes()],
            multi=True,
            placeholder="Select target variables",
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '20px'}),
    html.Hr(),
    html.Div([
        html.Button('Run Optimization', id='run-optimization-button', n_clicks=0, disabled=False)
    ], style={'textAlign': 'center'}),
    html.Br(),
    dcc.Loading(
        id="loading-results",
        type="circle",
        children=[
            html.Div(id='progress-messages', style={'whiteSpace': 'pre-line', 'textAlign': 'center'}),
            html.Div(id='optimization-results')
        ]
    ),
    dcc.Store(id='stored-network'),
    dcc.Store(id='uploaded-bif-content'),
    dcc.Interval(
        id='progress-interval',
        interval=1*1000,  # 1 second
        n_intervals=0,
        disabled=True
    )
])

from dash.dependencies import ALL

# Callback to update evidence values dropdowns based on selected evidence variables
@app.callback(
    Output('evidence-values-container', 'children'),
    Input('evidence-vars-dropdown', 'value'),
    State('stored-network', 'data'),
    State('uploaded-bif-content', 'data')
)
def update_evidence_values(evidence_vars, stored_network, uploaded_bif_content):
    if not evidence_vars:
        return []
    model = get_model(stored_network, uploaded_bif_content)
    children = []
    for var in evidence_vars:
        # Get the possible states for the variable
        var_states = model.get_cpds(var).state_names[var]
        children.append(html.Div([
            html.Label(f'Select value for {var}', style={'textAlign': 'center'}),
            dcc.Dropdown(
                id={'type': 'evidence-value-dropdown', 'index': var},
                options=[{'label': state, 'value': state} for state in var_states],
                value=var_states[0],  # Default to the first state
                style={'width': '50%', 'margin': '0 auto'}
            )
        ], style={'margin-bottom': '10px'}))
    return children

# Callback to update target variables options based on selected evidence variables
@app.callback(
    Output('target-vars-dropdown', 'options'),
    Input('evidence-vars-dropdown', 'value'),
    State('stored-network', 'data'),
    State('uploaded-bif-content', 'data')
)
def update_target_options(evidence_vars, stored_network, uploaded_bif_content):
    model = get_model(stored_network, uploaded_bif_content)
    if not evidence_vars:
        return [{'label': var, 'value': var} for var in model.nodes()]
    else:
        available_vars = [var for var in model.nodes() if var not in evidence_vars]
        return [{'label': var, 'value': var} for var in available_vars]

# Helper function to get the model
def get_model(stored_network, uploaded_bif_content):
    if stored_network is None:
        # Use default model
        return default_model
    try:
        if stored_network['network_type'] == 'path':
            model = reader.get_model()
            logger.info(f"Using default network: {stored_network['network_name']}")
            return model
        elif stored_network['network_type'] == 'string':
            reader = BIFReader(string=stored_network['content'])
            model = reader.get_model()
            logger.info(f"Using uploaded network: {stored_network['network_name']}")
            return model
        else:
            logger.error("Invalid network type")
            return default_model
    except Exception as e:
        logger.error(f"Error loading network: {e}")
        return default_model

@app.callback(
    Output('stored-network', 'data'),
    Input('upload-bif', 'contents'),
    State('upload-bif', 'filename'),
    Input('use-default-network', 'value')
)
def load_network(contents, filename, use_default_value):
    if 'default' in use_default_value or contents is None:
        # Use default network
        logger.info("Loaded default network: asia.bif")
        return {'network_name': 'asia.bif', 'network_type': 'path', 'content': 'asia.bif'}
    else:
        # Load network from uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            bif_data = decoded.decode('utf-8')
            # Try to load the network to ensure it's valid
            reader = BIFReader(string=bif_data)
            model = reader.get_model()
            logger.info(f"Loaded network from uploaded file: {filename}")
            # Store the content
            return {'network_name': filename, 'network_type': 'string', 'content': bif_data}
        except Exception as e:
            logger.error(f"Error loading network: {e}")
            return dash.no_update

@app.callback(
    Output('optimization-results', 'children'),
    Output('progress-interval', 'disabled'),
    Output('run-optimization-button', 'disabled'),
    Input('run-optimization-button', 'n_clicks'),
    State('stored-network', 'data'),
    State('uploaded-bif-content', 'data'),
    State('evidence-vars-dropdown', 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'id'),
    State('target-vars-dropdown', 'value')
)
def run_optimization(n_clicks, stored_network, uploaded_bif_content,
                     evidence_vars, evidence_values, evidence_ids, target_vars):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    if stored_network is None:
        return html.Div("Please upload a .bif file or select to use the default network.", style={'color': 'red', 'textAlign': 'center'}), True, False

    # Disable the Run button and enable the Interval component
    interval_disabled = False
    run_button_disabled = True

    model = get_model(stored_network, uploaded_bif_content)

    if not evidence_vars or not target_vars:
        return html.Div("Please select both evidence and target variables.", style={'color': 'red', 'textAlign': 'center'}), True, False

    # Ensure no variables are selected as both evidence and target
    overlap = set(evidence_vars) & set(target_vars)
    if overlap:
        return html.Div(f"The following variables cannot be both evidence and target: {', '.join(overlap)}", style={'color': 'red', 'textAlign': 'center'}), True, False

    # Build the evidence dictionary
    evidence = {}
    for var_id, value in zip(evidence_ids, evidence_values):
        var = var_id['index']
        evidence[var] = value

    # Validate evidence values
    for var in evidence_vars:
        var_states = model.get_cpds(var).state_names[var]
        if evidence[var] not in var_states:
            return html.Div(f"Invalid value '{evidence[var]}' for variable '{var}'.", style={'color': 'red', 'textAlign': 'center'}), True, False

    logger.info(f"Evidence: {evidence}")
    logger.info(f"Target Variables: {target_vars}")

    # Run the algorithms
    results = run_all_algorithms(model, evidence, target_vars)

    # Disable the Interval component after completion
    interval_disabled = True
    run_button_disabled = False

    # Format the results to display
    table_header = [
        html.Thead(html.Tr([
            html.Th('Algorithm'),
            html.Th('Solution'),
            html.Th('GBF'),
            html.Th('Time (sec)')
        ]))
    ]
    table_rows = []
    for res in results:
        solution_str = json.dumps(res['Solution']) if isinstance(res['Solution'], dict) else str(res['Solution'])
        row = html.Tr([
            html.Td(res['Algorithm']),
            html.Td(solution_str),
            html.Td(f"{res['GBF']:.4f}" if isinstance(res['GBF'], (int, float)) else res['GBF']),
            html.Td(f"{res['Time']:.2f}")
        ])
        table_rows.append(row)
    table_body = [html.Tbody(table_rows)]
    table = html.Table(table_header + table_body, style={
        'width': '80%',
        'margin': '0 auto',
        'textAlign': 'center',
        'border': '1px solid black'
    })

    return table, interval_disabled, run_button_disabled

@app.callback(
    Output('progress-messages', 'children'),
    Input('progress-interval', 'n_intervals')
)
def update_progress_messages(n):
    with progress_lock:
        if progress_messages:
            messages = "\n".join(progress_messages)
            return messages
        else:
            return ''

def update_progress(message):
    with progress_lock:
        progress_messages.append(message)

def run_all_algorithms(model, evidence, target_vars):
    results = []
    logger.info("Starting run_all_algorithms")

    # Clear previous progress messages
    with progress_lock:
        progress_messages.clear()

    # Adjusted parameters for faster execution
    size_gen = 10
    dead_iter = 5
    pop_size = 10
    n_gen = 10
    max_steps = 100

    # UMDAcat_mre2
    try:
        update_progress("Running UMDAcat_mre2")
        s = time.time()
        sol, gbf, _ = mre.UMDAcat_mre2(
            model,
            evidence,
            target_vars,
            size_gen=size_gen,
            dead_iter=dead_iter,
            verbose=False,
            alpha=0.8,
            best_init=True
        )
        e = time.time()
        results.append({'Algorithm': 'UMDAcat_mre2', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("UMDAcat_mre2 completed successfully")
    except Exception as ex:
        update_progress(f"UMDAcat_mre2 failed: {ex}")
        results.append({'Algorithm': 'UMDAcat_mre2', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # Repeat similar blocks for other algorithms, updating progress messages accordingly
    # DEA MRE
    try:
        update_progress("Running DEA MRE")
        s = time.time()
        sol, gbf = mre.dea_mre(model, evidence, target_vars, pop_size, max_steps)
        e = time.time()
        results.append({'Algorithm': 'DEA MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("DEA MRE completed successfully")
    except Exception as ex:
        update_progress(f"DEA MRE failed: {ex}")
        results.append({'Algorithm': 'DEA MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # EBNA MRE
    try:
        update_progress("Running EBNA MRE")
        s = time.time()
        sol, gbf, _ = mre.ebna_mre(
            model,
            evidence,
            target_vars,
            size_gen=size_gen,
            dead_iter=dead_iter,
            verbose=False,
            alpha=0.8,
            best_init=True
        )
        e = time.time()
        results.append({'Algorithm': 'EBNA MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("EBNA MRE completed successfully")
    except Exception as ex:
        update_progress(f"EBNA MRE failed: {ex}")
        results.append({'Algorithm': 'EBNA MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # ES MRE
    try:
        update_progress("Running ES MRE")
        s = time.time()
        sol, gbf = mre.es_mre(model, evidence, target_vars, pop_size, max_steps)
        e = time.time()
        results.append({'Algorithm': 'ES MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("ES MRE completed successfully")
    except Exception as ex:
        update_progress(f"ES MRE failed: {ex}")
        results.append({'Algorithm': 'ES MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # GA MRE
    try:
        update_progress("Running GA MRE")
        s = time.time()
        sol, gbf = mre.ga_mre(model, evidence, target_vars, pop_size, max_steps)
        e = time.time()
        results.append({'Algorithm': 'GA MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("GA MRE completed successfully")
    except Exception as ex:
        update_progress(f"GA MRE failed: {ex}")
        results.append({'Algorithm': 'GA MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # Hierarchical Beam Search
    try:
        update_progress("Running Hierarchical Beam Search")
        s = time.time()
        sol, gbf = mre.hierarchical_beam_search(
            model,
            evidence,
            target_vars,
            width=5,        # Corrected parameter name
            delta=1+1e-8,
            max_steps=10,
            k=2
        )
        e = time.time()
        results.append({'Algorithm': 'Hierarchical Beam Search', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("Hierarchical Beam Search completed successfully")
    except Exception as ex:
        update_progress(f"Hierarchical Beam Search failed: {ex}")
        results.append({'Algorithm': 'Hierarchical Beam Search', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # NSGA2 MRE
    try:
        update_progress("Running NSGA2 MRE")
        s = time.time()
        sol, gbf = mre.nsga2_mre(model, evidence, target_vars, pop_size=pop_size, n_gen=n_gen, best_init=True, period=10)
        e = time.time()
        results.append({'Algorithm': 'NSGA2 MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("NSGA2 MRE completed successfully")
    except Exception as ex:
        update_progress(f"NSGA2 MRE failed: {ex}")
        results.append({'Algorithm': 'NSGA2 MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # PSO MRE
    try:
        update_progress("Running PSO MRE")
        s = time.time()
        sol, gbf = mre.pso_mre(model, evidence, target_vars, pop_size, max_steps)
        e = time.time()
        results.append({'Algorithm': 'PSO MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("PSO MRE completed successfully")
    except Exception as ex:
        update_progress(f"PSO MRE failed: {ex}")
        results.append({'Algorithm': 'PSO MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    # Tabu MRE
    try:
        update_progress("Running Tabu MRE")
        s = time.time()
        sol, gbf = mre.tabu_mre(
            model,
            evidence,
            target_vars,
            max_steps=max_steps,
            tabu_size=30,
            more_targets=1
        )
        e = time.time()
        results.append({'Algorithm': 'Tabu MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
        update_progress("Tabu MRE completed successfully")
    except Exception as ex:
        update_progress(f"Tabu MRE failed: {ex}")
        results.append({'Algorithm': 'Tabu MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

    update_progress("Finished run_all_algorithms")
    return results

if __name__ == '__main__':
        app.run_server(debug=True, host='0.0.0.0', port=8052)
