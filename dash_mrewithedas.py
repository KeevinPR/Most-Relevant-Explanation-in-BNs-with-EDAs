import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import time
import json
import logging
import base64
import threading
from dash.exceptions import PreventUpdate
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
import warnings

import mre  

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix='/Evidence/MREWithEDAsDash/',
    suppress_callback_exceptions=True
)
server = app.server

progress_lock = threading.Lock()
progress_messages = []

# IMPORTANT: We do NOT load asia.bif by default. Instead, we store None here:
default_model = None

# We define the minimum number of target variables required by each algorithm:
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

app.layout = html.Div([
    dcc.Loading(
        id="global-spinner",
        overlay_style={"visibility": "visible", "filter": "blur(1px)"},
        type="circle",
        fullscreen=False,
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
                html.H3("1. Load Bayesian Network (.bif)", style={'textAlign': 'center'}),
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
                ], style={'textAlign': 'center'}),
            ]),

            html.Div(id='upload-status', style={'textAlign': 'center', 'color': 'green'}),

            # (B) Evidence and Targets
            html.Div(className="card", children=[
                html.H3("2. Select Evidence", style={'textAlign': 'center'}),
                # This dropdown will be dynamically populated once a model is selected
                dcc.Dropdown(
                    id='evidence-vars-dropdown',
                    options=[],
                    multi=True,
                    placeholder="Select evidence variables",
                    style={'width': '50%', 'margin': '0 auto'}
                ),
                html.Div(id='evidence-values-container')
            ]),

            html.Div(className="card", children=[
                html.H3("3. Select Target Variables", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='target-vars-dropdown',
                    options=[],
                    multi=True,
                    placeholder="Select target variables",
                    style={'width': '50%', 'margin': '0 auto'}
                ),
                html.Div(
                    'Note: Some algorithms need at least 2 target variables (UMDAcat_mre2, GA MRE, etc.)',
                    style={'textAlign': 'center', 'fontSize': '12px', 'color': 'gray', 'marginTop': '10px'}
                ),
            ]),

            # (C) Algorithm parameters
            html.Div(className="card", children=[
                html.H3("4. Algorithm Parameters", style={'textAlign': 'center'}),
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
            ]),

            # (D) "Run" button + progress messages
            html.Div([
                dcc.Loading(
                    id='loading-run-button',
                    type='circle',
                    children=[
                        html.Div([
                            html.Button(
                                'Run Optimization',
                                id='run-optimization-button',
                                n_clicks=0,
                                style={'margin': '10px'}
                            )
                        ], style={'textAlign': 'center'}),
                        dcc.Store(id='run-button-store')
                    ]
                )
            ], id='run-button-container', style={'textAlign': 'center'}),

            html.Br(),
            html.Div(id='progress-messages', style={'whiteSpace': 'pre-line', 'textAlign': 'center'}),
            html.Div(id='optimization-results'),

            # (E) Interval for progress tracking, plus hidden stores
            html.Div(id='scroll-helper', style={'display': 'none'}),
            dcc.Store(id='stored-network'),
            dcc.Store(id='uploaded-bif-content'),
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
    Input('upload-bif', 'contents'),
    State('upload-bif', 'filename'),
    Input('use-default-network', 'value')
)
def load_network(contents, filename, use_default_value):
    """
    1) If user checks 'default', load asia.bif from disk.
    2) If user uploads a .bif, parse that.
    3) Otherwise, no network is used (None).
    """
    global cached_model
    cached_model = None
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
                msg
            )
        except Exception as e:
            logger.error(f"Error reading default network: {e}")
            return None, f"Error reading default network: {e}"

    if contents is not None:
        # The user uploaded a file
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
                msg
            )
        except Exception as e:
            logger.error(f"Error loading network from {filename}: {e}")
            return None, f"Error loading {filename}: {e}"

    # If neither default nor file upload
    return None, "No network selected. Upload a file or check the default option."

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
    Output('evidence-vars-dropdown', 'options'),
    Input('stored-network', 'data')
)
def update_evidence_variables(stored_network):
    m = get_model(stored_network)
    if not m:
        return []
    return [{'label': var, 'value': var} for var in m.nodes()]

# Build the dynamic evidence-value dropdowns
@app.callback(
    Output('evidence-values-container', 'children'),
    Input('evidence-vars-dropdown', 'value'),
    State('stored-network', 'data')
)
def update_evidence_values(evidence_vars, stored_network):
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
    Output('target-vars-dropdown', 'options'),
    Input('evidence-vars-dropdown', 'value'),
    State('stored-network', 'data')
)
def update_target_options(evidence_vars, stored_network):
    m = get_model(stored_network)
    if m is None:
        return []
    all_vars = set(m.nodes())
    evidence_vars = evidence_vars or []
    available = [v for v in all_vars if v not in evidence_vars]

    valid_targets = []
    for v in available:
        states = m.get_cpds(v).state_names[v]
        if len(states) > 1:
            valid_targets.append(v)

    return [{'label': x, 'value': x} for x in valid_targets]

# Callback to run optimization
@app.callback(
    Output('optimization-results', 'children'),
    Output('progress-interval', 'disabled'),
    Output('run-button-store', 'data'),
    Input('run-optimization-button', 'n_clicks'),
    State('stored-network', 'data'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'id'),
    State('target-vars-dropdown', 'value'),
    State('pop-size-input', 'value'),
    State('num-gen-input', 'value'),
    State('max-steps-input', 'value'),
    State('dead-iter-input', 'value')
)
def run_optimization(n_clicks,
                     stored_network,
                     evidence_values, evidence_ids,
                     target_vars,
                     pop_size, n_gen, max_steps, dead_iter):
    if not n_clicks:
        raise PreventUpdate

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

    # Run the set of algorithms
    results = run_all_algorithms(model, evidence_dict, final_targets, pop_size, n_gen, max_steps, dead_iter)

    # Once finished, disable interval and show the run button again
    interval_disabled = True
    hide_run_button = False

    # Format the results table
    header = html.Thead(html.Tr([
        html.Th("Algorithm"),
        html.Th("Solution"),
        html.Th("GBF"),
        html.Th("Time (s)")
    ]))
    rows = []
    for r in results:
        sol_str = json.dumps(r['Solution']) if isinstance(r['Solution'], dict) else str(r['Solution'])
        gbf_str = f"{r['GBF']:.4f}" if isinstance(r['GBF'], (float, int)) else str(r['GBF'])
        t_str = f"{r['Time']:.2f}"
        row = html.Tr([
            html.Td(r['Algorithm']),
            html.Td(sol_str),
            html.Td(gbf_str),
            html.Td(t_str)
        ])
        rows.append(row)

    body = html.Tbody(rows)
    table = html.Table([header, body], style={
        'width': '80%',
        'margin': '0 auto',
        'textAlign': 'center',
        'border': '1px solid black'
    })

    return table, interval_disabled, hide_run_button

@app.callback(
    Output('run-optimization-button', 'style'),
    Input('run-button-store', 'data')
)
def hide_run_button_callback(flag):
    if flag:
        return {'display': 'none'}
    return {'display': 'inline-block'}

@app.callback(
    Output('progress-messages', 'children'),
    Input('progress-interval', 'n_intervals'),
    State('progress-messages', 'children')
)
def update_progress_messages(n_intervals, existing):
    with progress_lock:
        if progress_messages:
            text = "\n".join(progress_messages)
            progress_messages.clear()
            return text
        else:
            return existing

def update_progress(msg):
    with progress_lock:
        progress_messages.append(msg)

def run_all_algorithms(model, evidence, targets, pop_size, n_gen, max_steps, dead_iter):
    logger.info("Starting run_all_algorithms")
    update_progress("Starting run_all_algorithms")

    with progress_lock:
        progress_messages.clear()

    results = []

    # Build a list of algorithms that have enough target variables
    algorithms = []
    for a, min_t in algorithm_requirements.items():
        if len(targets) >= min_t:
            algorithms.append(a)
        else:
            update_progress(f"Skipping {a}; requires >= {min_t} targets.")

    if not targets:
        update_progress("No targets found to optimize.")
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
                    more_targets=1
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("UMDAcat_mre2 completed successfully.")

            elif alg == 'DEA MRE':
                update_progress("Running DEA MRE")
                start = time.time()
                sol, gbf = mre.dea_mre(model, evidence, targets, pop_size, max_steps, more_targets=1)
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
                    more_targets=1
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("EBNA MRE completed successfully.")

            elif alg == 'ES MRE':
                update_progress("Running ES MRE")
                start = time.time()
                sol, gbf = mre.es_mre(model, evidence, targets, pop_size, max_steps, more_targets=1)
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("ES MRE completed successfully.")

            elif alg == 'GA MRE':
                update_progress("Running GA MRE")
                start = time.time()
                sol, gbf = mre.ga_mre(model, evidence, targets, pop_size, max_steps, more_targets=1)
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
                    more_targets=1
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
                    best_init=True, period=10, more_targets=1
                )
                elapsed = time.time() - start
                results.append({'Algorithm': alg, 'Solution': sol, 'GBF': gbf, 'Time': elapsed})
                update_progress("NSGA2 MRE completed successfully.")

            elif alg == 'PSO MRE':
                update_progress("Running PSO MRE")
                start = time.time()
                sol, gbf = mre.pso_mre(model, evidence, targets, pop_size, max_steps, more_targets=1)
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
                    more_targets=1
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

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8052)