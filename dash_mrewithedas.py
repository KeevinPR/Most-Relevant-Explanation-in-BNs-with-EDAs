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

#
import mre 

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#Dash app
app = dash.Dash(
    __name__,
    requests_pathname_prefix='/Evidence/MREWithEDAsDash/',
    suppress_callback_exceptions=True
)

progress_lock = threading.Lock()
progress_messages = []

# Loading the default Bayesian Network model
reader = BIFReader('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/MREWithEDAs/asia.bif')  # Ensure 'asia.bif' is in the same directory
default_model = reader.get_model()

# Algorithm requirements mapping
algorithm_requirements = {
    'UMDAcat_mre2': 2,
    'DEA MRE': 2,
    'EBNA MRE': 2,
    'ES MRE': 2,
    'GA MRE': 2,
    'Hierarchical Beam Search': 1,
    'NSGA2 MRE': 2,
    'PSO MRE': 2,
    'Tabu MRE': 2
}

app.layout = dcc.Loading(
    id="global-spinner",
    overlay_style={"visibility":"visible", "filter": "blur(1px)"},
    type="circle",        # "circle", "dot", "default"
    fullscreen=False,      # This ensures it covers the entire page
    children=html.Div([
    html.H1("Bayesian Network Optimization", style={'textAlign': 'center'}),
    
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
            options=[],
            multi=True,
            placeholder="Select target variables",
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '20px'}),
    html.Div(
        'Note: The following algorithms require at least 2 target variables: '
        'UMDAcat_mre2, DEA MRE, EBNA MRE, ES MRE, GA MRE, NSGA2 MRE, PSO MRE, Tabu MRE.',
        style={'textAlign': 'center','fontSize': '12px', 'color': 'gray'}
    ),
    html.Hr(),
    # Algorithm parameters
    html.Div([
        html.H3("Algorithm Parameters", style={'textAlign': 'center'}),
        html.Div([
            html.Label('Population Size:', style={'marginRight': '10px'}),
            dcc.Input(id='pop-size-input', type='number', value=10, min=1, step=1, style={'width': '60px'}),
            html.Label('Number of Generations:', style={'marginLeft': '20px', 'marginRight': '10px'}),
            dcc.Input(id='num-gen-input', type='number', value=10, min=1, step=1, style={'width': '60px'}),
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),
        html.Div([
            html.Label('Max Steps:', style={'marginRight': '10px'}),
            dcc.Input(id='max-steps-input', type='number', value=100, min=1, step=1, style={'width': '60px'}),
            html.Label('Dead Iterations:', style={'marginLeft': '20px', 'marginRight': '10px'}),
            dcc.Input(id='dead-iter-input', type='number', value=5, min=1, step=1, style={'width': '60px'}),
        ], style={'textAlign': 'center'}),
    ], style={'marginBottom': '20px'}),
    html.Hr(),
    html.Div([
        dcc.Loading(
            id='loading-run-button',
            type='circle',
            children=[
                html.Div([
                    html.Button('Run Optimization', id='run-optimization-button', n_clicks=0)
                ], style={'textAlign': 'center'}),
                dcc.Store(id='run-button-store')
            ]
        )
    ], id='run-button-container', style={'textAlign': 'center'}),
    html.Br(),
    html.Div(id='progress-messages', style={'whiteSpace': 'pre-line', 'textAlign': 'center'}),
    html.Div(id='optimization-results'),
    # Hidden Div for clientside callback output
    html.Div(id='scroll-helper', style={'display': 'none'}),
    
    dcc.Store(id='stored-network'),
    dcc.Store(id='uploaded-bif-content'),
    dcc.Interval(
        id='progress-interval',
        interval=1*1000,  # 1 second
        n_intervals=0,
        disabled=True
    )
])
)
#For scrolling down animation
app.clientside_callback(
    """
    function(results) {
        if (results) {
            setTimeout(function() {
                var elem = document.getElementById('optimization-results');
                console.log('Attempting to scroll after delay. Element:', elem);
                if (elem) {
                    elem.scrollIntoView({behavior: 'smooth'});
                } else {
                    console.log('Element not found after delay.');
                }
            }, 100); // Delay of 100 milliseconds
        }
        return '';
    }
    """,
    Output('scroll-helper', 'children'),
    [Input('optimization-results', 'children')]
)

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
        children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                f'Select value for {var}',
                                style={
                                    'width': '40%',
                                    'textAlign': 'right',
                                    'paddingRight': '10px'
                                }
                            ),
                            dcc.Dropdown(
                                id={'type': 'evidence-value-dropdown', 'index': var},
                                options=[{'label': state, 'value': state} for state in var_states],
                                value=var_states[0],  # Default to the first state
                                style={'width': '60%'}
                            )
                        ],
                        style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'width': '100%',
                            'margin': '0 auto'
                        }
                    )
                ],
                style={'marginBottom': '10px', 'width': '50%', 'margin': '0 auto'}
            )
        )
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
    all_vars = set(model.nodes())
    if not evidence_vars:
        evidence_vars = []
    # Exclude evidence variables and variables with only one state
    available_vars = [var for var in all_vars if var not in evidence_vars]
    valid_target_vars = []
    for var in available_vars:
        var_states = model.get_cpds(var).state_names[var]
        if len(var_states) >= 2:
            valid_target_vars.append(var)
    return [{'label': var, 'value': var} for var in valid_target_vars]

# Global variable to cache the model
cached_model = None

def get_model(stored_network, uploaded_bif_content):
    global cached_model
    if cached_model is not None:
        return cached_model
    if stored_network is None:
        # Use default model
        cached_model = default_model
        return default_model
    try:
        if stored_network['network_type'] == 'path':
            reader = BIFReader('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/MREWithEDAs/asia.bif')
            model = reader.get_model()
            cached_model = model
            logger.info(f"Using default network: {stored_network['network_name']}")
            return model
        elif stored_network['network_type'] == 'string':
            reader = BIFReader(string=stored_network['content'])
            model = reader.get_model()
            cached_model = model
            logger.info(f"Using uploaded network: {stored_network['network_name']}")
            return model
        else:
            logger.error("Invalid network type")
            cached_model = default_model
            return default_model
    except Exception as e:
        logger.error(f"Error loading network: {e}")
        cached_model = default_model
        return default_model

@app.callback(
    Output('stored-network', 'data'),
    Input('upload-bif', 'contents'),
    State('upload-bif', 'filename'),
    Input('use-default-network', 'value')
)
def load_network(contents, filename, use_default_value):
    global cached_model
    cached_model = None  # Reset cached model when network changes
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
    Output('run-button-store', 'data'),
    Input('run-optimization-button', 'n_clicks'),
    State('stored-network', 'data'),
    State('uploaded-bif-content', 'data'),
    State('evidence-vars-dropdown', 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'id'),
    State('target-vars-dropdown', 'value'),
    State('pop-size-input', 'value'),
    State('num-gen-input', 'value'),
    State('max-steps-input', 'value'),
    State('dead-iter-input', 'value')
)
def run_optimization(n_clicks, stored_network, uploaded_bif_content,
                     evidence_vars, evidence_values, evidence_ids, target_vars,
                     pop_size_input, num_gen_input, max_steps_input, dead_iter_input):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    if stored_network is None:
        return html.Div("Please upload a .bif file or select to use the default network.",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # Disable the Run button and enable the Interval component
    interval_disabled = False
    run_button_hidden = True  # We'll use this to hide the button

    model = get_model(stored_network, uploaded_bif_content)

    if not evidence_vars or not target_vars:
        return html.Div("Please select both evidence and target variables.",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # Ensure no variables are selected as both evidence and target
    overlap = set(evidence_vars) & set(target_vars)
    if overlap:
        return html.Div(f"The following variables cannot be both evidence and target: {', '.join(overlap)}",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    # Build the evidence dictionary
    evidence = {}
    for var_id, value in zip(evidence_ids, evidence_values):
        var = var_id['index']
        evidence[var] = value

    # Validate evidence values
    for var in evidence_vars:
        var_states = model.get_cpds(var).state_names[var]
        if evidence[var] not in var_states:
            return html.Div(f"Invalid value '{evidence[var]}' for variable '{var}'.",
                            style={'color': 'red', 'textAlign': 'center'}), True, False

    # Check target variables for at least two possible states given the evidence
    inference = VariableElimination(model)
    valid_target_vars = []
    epsilon = 1e-8  # Tolerance for floating-point precision
    for var in target_vars:
        try:
            marginal = inference.query([var], evidence=evidence, show_progress=False)
            probs = marginal.values.flatten()
            max_prob = max(probs)
            if max_prob >= 1 - epsilon:
                update_progress(f"Variable {var} is deterministic given the evidence and will be excluded from target variables.")
            else:
                valid_target_vars.append(var)
        except Exception as e:
            update_progress(f"Failed to compute marginal for variable {var}: {e}")

    if not valid_target_vars:
        return html.Div("No valid target variables that are not deterministic given the evidence.",
                        style={'color': 'red', 'textAlign': 'center'}), True, False

    logger.info(f"Evidence: {evidence}")
    logger.info(f"Target Variables: {valid_target_vars}")

    # Run the algorithms with the provided parameters
    results = run_all_algorithms(model, evidence, valid_target_vars,
                                 pop_size_input, num_gen_input, max_steps_input, dead_iter_input)

    # Disable the Interval component after completion
    interval_disabled = True
    run_button_hidden = False  # Show the button again

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

    return table, interval_disabled, run_button_hidden

# Callback to hide/show the run button based on the store
@app.callback(
    Output('run-optimization-button', 'style'),
    Input('run-button-store', 'data')
)
def hide_run_button(run_button_hidden):
    if run_button_hidden:
        return {'display': 'none'}
    else:
        return {'display': 'inline-block'}

@app.callback(
    Output('progress-messages', 'children'),
    Input('progress-interval', 'n_intervals'),
    State('progress-messages', 'children')
)
def update_progress_messages(n, existing_messages):
    with progress_lock:
        if progress_messages:
            messages = "\n".join(progress_messages)
            # Clear messages after displaying
            progress_messages.clear()
            return messages
        else:
            return existing_messages

def update_progress(message):
    with progress_lock:
        progress_messages.append(message)

def run_all_algorithms(model, evidence, target_vars, pop_size, n_gen, max_steps, dead_iter):
    results = []
    logger.info("Starting run_all_algorithms")
    update_progress("Starting run_all_algorithms")

    # Clear previous progress messages
    with progress_lock:
        progress_messages.clear()

    # Adjusted parameters based on user input
    size_gen = pop_size
    dead_iter = dead_iter
    pop_size = pop_size
    n_gen = n_gen
    max_steps = max_steps
    
    # List of algorithms to run
    algorithms_to_run = []
    for algorithm, min_targets in algorithm_requirements.items():
        if len(target_vars) >= min_targets:
            algorithms_to_run.append(algorithm)
        else:
            update_progress(f"Skipping {algorithm} due to insufficient target variables (requires at least {min_targets})")

    # Proceed to run algorithms only if target_vars is not empty
    if not target_vars:
        update_progress("No target variables to optimize.")
        return results

    # Now, run only the algorithms in algorithms_to_run
    for algorithm in algorithms_to_run:
        if algorithm == 'UMDAcat_mre2':
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
                    best_init=True,
                    more_targets=1  # Adjusted parameter
                )
                e = time.time()
                results.append({'Algorithm': 'UMDAcat_mre2', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("UMDAcat_mre2 completed successfully")
            except Exception as ex:
                update_progress(f"UMDAcat_mre2 failed: {ex}")
                results.append({'Algorithm': 'UMDAcat_mre2', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'DEA MRE':
            # DEA MRE
            try:
                update_progress("Running DEA MRE")
                s = time.time()
                sol, gbf = mre.dea_mre(model, evidence, target_vars, pop_size, max_steps, more_targets=1)  # Adjusted parameter
                e = time.time()
                results.append({'Algorithm': 'DEA MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("DEA MRE completed successfully")
            except Exception as ex:
                update_progress(f"DEA MRE failed: {ex}")
                results.append({'Algorithm': 'DEA MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'EBNA MRE':
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
                    best_init=True,
                    more_targets=1  # Adjusted parameter
                )
                e = time.time()
                results.append({'Algorithm': 'EBNA MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("EBNA MRE completed successfully")
            except Exception as ex:
                update_progress(f"EBNA MRE failed: {ex}")
                results.append({'Algorithm': 'EBNA MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'ES MRE':
            # ES MRE
            try:
                update_progress("Running ES MRE")
                s = time.time()
                sol, gbf = mre.es_mre(model, evidence, target_vars, pop_size, max_steps, more_targets=1)  # Adjusted parameter
                e = time.time()
                results.append({'Algorithm': 'ES MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("ES MRE completed successfully")
            except Exception as ex:
                update_progress(f"ES MRE failed: {ex}")
                results.append({'Algorithm': 'ES MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'GA MRE':
            # GA MRE
            try:
                update_progress("Running GA MRE")
                s = time.time()
                sol, gbf = mre.ga_mre(model, evidence, target_vars, pop_size, max_steps, more_targets=1)  # Adjusted parameter
                e = time.time()
                results.append({'Algorithm': 'GA MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("GA MRE completed successfully")
            except Exception as ex:
                update_progress(f"GA MRE failed: {ex}")
                results.append({'Algorithm': 'GA MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'Hierarchical Beam Search':
            # Hierarchical Beam Search
            try:
                update_progress("Running Hierarchical Beam Search")
                s = time.time()
                sol, gbf = mre.hierarchical_beam_search(
                    model,
                    evidence,
                    target_vars,
                    5,         # beam_width or appropriate parameter
                    1+1e-8,    # delta
                    10,        # max_steps
                    2,
                    more_targets=1  # Adjusted parameter
                )
                e = time.time()
                results.append({'Algorithm': 'Hierarchical Beam Search', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("Hierarchical Beam Search completed successfully")
            except Exception as ex:
                update_progress(f"Hierarchical Beam Search failed: {ex}")
                results.append({'Algorithm': 'Hierarchical Beam Search', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'NSGA2 MRE':
            # NSGA2 MRE
            try:
                update_progress("Running NSGA2 MRE")
                s = time.time()
                sol, gbf = mre.nsga2_mre(model, evidence, target_vars, pop_size=pop_size, n_gen=n_gen, best_init=True, period=10, more_targets=1)  # Adjusted parameter
                e = time.time()
                results.append({'Algorithm': 'NSGA2 MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("NSGA2 MRE completed successfully")
            except Exception as ex:
                update_progress(f"NSGA2 MRE failed: {ex}")
                results.append({'Algorithm': 'NSGA2 MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'PSO MRE':
            # PSO MRE
            try:
                update_progress("Running PSO MRE")
                s = time.time()
                sol, gbf = mre.pso_mre(model, evidence, target_vars, pop_size, max_steps, more_targets=1)  # Adjusted parameter
                e = time.time()
                results.append({'Algorithm': 'PSO MRE', 'Solution': sol, 'GBF': gbf, 'Time': e - s})
                update_progress("PSO MRE completed successfully")
            except Exception as ex:
                update_progress(f"PSO MRE failed: {ex}")
                results.append({'Algorithm': 'PSO MRE', 'Solution': str(ex), 'GBF': 'Error', 'Time': 0})

        elif algorithm == 'Tabu MRE':
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
                    more_targets=1  # Adjusted parameter
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
