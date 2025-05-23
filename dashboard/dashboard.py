# Phase 5: Visualization Dashboard for DeFi Credit Risk

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import pickle
import sys
import json
from datetime import datetime
import base64
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.credit_risk_model import CreditRiskModel
from ml_models.portfolio_optimizer import PortfolioOptimizer

class DeFiDashboard:
    """
    Interactive Dashboard for DeFi Credit Risk Analysis and Portfolio Optimization.
    
    This dashboard integrates the simulation environment, machine learning models,
    and portfolio optimization components into a unified interface.
    """
    
    def __init__(self):
        """Initialize the dashboard with data and models."""
        # Set up paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.ml_dir = os.path.join(self.base_dir, 'ml_models')
        self.portfolio_dir = os.path.join(self.data_dir, 'portfolio_results')
        self.dashboard_assets = os.path.join(self.base_dir, 'dashboard', 'assets')
        
        # Create directories if they don't exist
        os.makedirs(self.dashboard_assets, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Initialize models
        self.credit_model = self.load_credit_model()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            assets_folder=self.dashboard_assets,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        self.app.title = "DeFi Credit Risk Dashboard"
        
        # Set up app layout
        self.setup_layout()
        
        # Set up callbacks
        self.setup_callbacks()
    
    def load_data(self):
        """Load all necessary data for the dashboard."""
        try:
            # Load loan data
            loan_file = os.path.join(self.data_dir, 'complete_loan_data.csv')
            self.loan_data = pd.read_csv(loan_file)
            print(f"Loaded loan data with {self.loan_data.shape[0]} rows.")
            
            # Load borrower data
            borrower_file = os.path.join(self.data_dir, 'borrowers.csv')
            self.borrower_data = pd.read_csv(borrower_file)
            print(f"Loaded borrower data with {self.borrower_data.shape[0]} rows.")
            
            # Load portfolio optimization results
            portfolio_file = os.path.join(self.portfolio_dir, 'portfolio_optimization_results.pkl')
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'rb') as f:
                    self.portfolio_results = pickle.load(f)
                print("Loaded portfolio optimization results.")
            else:
                self.portfolio_results = None
                print("Portfolio optimization results not found.")
            
            # Load segment statistics
            segment_file = os.path.join(self.portfolio_dir, 'segment_statistics.csv')
            if os.path.exists(segment_file):
                self.segment_stats = pd.read_csv(segment_file)
                print(f"Loaded segment statistics with {self.segment_stats.shape[0]} rows.")
            else:
                self.segment_stats = None
                print("Segment statistics not found.")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            # Initialize with empty dataframes if loading fails
            self.loan_data = pd.DataFrame()
            self.borrower_data = pd.DataFrame()
            self.portfolio_results = None
            self.segment_stats = None
    
    def load_credit_model(self):
        """Load the trained credit risk model."""
        try:
            model_path = os.path.join(self.ml_dir, 'saved_models', 'credit_risk_model.pkl')
            if os.path.exists(model_path):
                model = CreditRiskModel()
                model.load_model()
                print("Loaded credit risk model.")
                return model
            else:
                print("Credit risk model not found.")
                return None
        except Exception as e:
            print(f"Error loading credit risk model: {e}")
            return None
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("DeFi Credit Risk Analysis Dashboard", className="header-title"),
                html.P("Autonomous Multi-Agent AI System for Credit Risk in DeFi", className="header-description"),
            ], className="header"),
            
            # Tabs
            dcc.Tabs([
                # Overview Tab
                dcc.Tab(label="Overview", children=[
                    html.Div([
                        html.H2("Project Overview"),
                        html.P("""
                            This dashboard provides an interactive interface for analyzing credit risk in DeFi lending.
                            It integrates simulation data, machine learning models, and portfolio optimization techniques
                            to help lenders make informed decisions about capital allocation.
                        """),
                        
                        html.Div([
                            html.Div([
                                html.H3("Key Metrics"),
                                html.Div(id="overview-metrics", className="metrics-container")
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Default Rate by Trust Score"),
                                dcc.Graph(id="overview-trust-default")
                            ], className="six columns"),
                        ], className="row"),
                        
                        html.Div([
                            html.Div([
                                html.H3("Loan Amount Distribution"),
                                dcc.Graph(id="overview-loan-amount")
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Default Rate by Loan Size"),
                                dcc.Graph(id="overview-size-default")
                            ], className="six columns"),
                        ], className="row"),
                    ], className="tab-content")
                ]),
                
                # Credit Risk Modeling Tab
                dcc.Tab(label="Credit Risk Modeling", children=[
                    html.Div([
                        html.H2("Credit Risk Model Analysis"),
                        
                        html.Div([
                            html.Div([
                                html.H3("Feature Importance"),
                                dcc.Graph(id="model-feature-importance")
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Model Performance"),
                                dcc.Graph(id="model-performance")
                            ], className="six columns"),
                        ], className="row"),
                        
                        html.Div([
                            html.H3("Default Probability Prediction"),
                            html.P("Enter loan details to predict default probability:"),
                            
                            html.Div([
                                html.Div([
                                    html.Label("Trust Score (0-1):"),
                                    dcc.Slider(
                                        id="pred-trust-score",
                                        min=0, max=1, step=0.01, value=0.7,
                                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'}
                                    ),
                                ], className="four columns"),
                                
                                html.Div([
                                    html.Label("Loan Amount ($):"),
                                    dcc.Input(
                                        id="pred-loan-amount",
                                        type="number",
                                        value=10000,
                                        min=1000,
                                        max=100000
                                    ),
                                ], className="four columns"),
                                
                                html.Div([
                                    html.Label("Interest Rate (%):"),
                                    dcc.Input(
                                        id="pred-interest-rate",
                                        type="number",
                                        value=5,
                                        min=1,
                                        max=20
                                    ),
                                ], className="four columns"),
                            ], className="row"),
                            
                            html.Div([
                                html.Div([
                                    html.Label("Collateral Ratio:"),
                                    dcc.Input(
                                        id="pred-collateral-ratio",
                                        type="number",
                                        value=1.5,
                                        min=1,
                                        max=3,
                                        step=0.1
                                    ),
                                ], className="four columns"),
                                
                                html.Div([
                                    html.Label("Market Volatility:"),
                                    dcc.Slider(
                                        id="pred-market-volatility",
                                        min=0.05, max=0.5, step=0.01, value=0.2,
                                        marks={0.05: '0.05', 0.2: '0.2', 0.35: '0.35', 0.5: '0.5'}
                                    ),
                                ], className="four columns"),
                                
                                html.Div([
                                    html.Button("Predict", id="predict-button", className="button"),
                                ], className="four columns", style={"margin-top": "25px"}),
                            ], className="row"),
                            
                            html.Div([
                                html.Div(id="prediction-result", className="prediction-result")
                            ], className="row", style={"margin-top": "20px"}),
                            
                            html.Div([
                                html.H4("Explanation"),
                                html.Div(id="prediction-explanation")
                            ], className="row", style={"margin-top": "20px"}),
                        ], className="prediction-container"),
                    ], className="tab-content")
                ]),
                
                # Portfolio Optimization Tab
                dcc.Tab(label="Portfolio Optimization", children=[
                    html.Div([
                        html.H2("Portfolio Optimization Analysis"),
                        
                        html.Div([
                            html.Div([
                                html.H3("Efficient Frontier"),
                                dcc.Graph(id="portfolio-efficient-frontier")
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Portfolio Allocation"),
                                dcc.Graph(id="portfolio-allocation")
                            ], className="six columns"),
                        ], className="row"),
                        
                        html.Div([
                            html.H3("Segment Performance"),
                            dcc.Graph(id="segment-performance")
                        ]),
                        
                        html.Div([
                            html.H3("Portfolio Optimizer"),
                            html.P("Adjust risk tolerance to see optimal portfolio allocation:"),
                            
                            html.Div([
                                html.Label("Risk Tolerance:"),
                                dcc.Slider(
                                    id="risk-tolerance",
                                    min=0.1, max=5, step=0.1, value=1,
                                    marks={0.1: 'Low', 1: 'Medium', 5: 'High'}
                                ),
                            ]),
                            
                            html.Div([
                                html.Button("Optimize", id="optimize-button", className="button"),
                            ], style={"margin-top": "15px"}),
                            
                            html.Div([
                                html.Div(id="optimization-result", className="optimization-result")
                            ], style={"margin-top": "20px"}),
                        ], className="optimization-container"),
                    ], className="tab-content")
                ]),
                
                # Simulation Tab
                dcc.Tab(label="Simulation", children=[
                    html.Div([
                        html.H2("Simulation Environment"),
                        
                        html.Div([
                            html.H3("Simulation Parameters"),
                            
                            html.Div([
                                html.Div([
                                    html.Label("Number of Borrowers:"),
                                    dcc.Input(
                                        id="sim-num-borrowers",
                                        type="number",
                                        value=100,
                                        min=10,
                                        max=1000
                                    ),
                                ], className="four columns"),
                                
                                html.Div([
                                    html.Label("Number of Lenders:"),
                                    dcc.Input(
                                        id="sim-num-lenders",
                                        type="number",
                                        value=10,
                                        min=1,
                                        max=100
                                    ),
                                ], className="four columns"),
                                
                                html.Div([
                                    html.Label("Simulation Duration (days):"),
                                    dcc.Input(
                                        id="sim-duration",
                                        type="number",
                                        value=365,
                                        min=30,
                                        max=3650
                                    ),
                                ], className="four columns"),
                            ], className="row"),
                            
                            html.Div([
                                html.Div([
                                    html.Label("Initial Market Volatility:"),
                                    dcc.Slider(
                                        id="sim-volatility",
                                        min=0.05, max=0.5, step=0.01, value=0.2,
                                        marks={0.05: '0.05', 0.2: '0.2', 0.35: '0.35', 0.5: '0.5'}
                                    ),
                                ], className="six columns"),
                                
                                html.Div([
                                    html.Label("Initial Interest Rate:"),
                                    dcc.Slider(
                                        id="sim-interest-rate",
                                        min=0.01, max=0.2, step=0.01, value=0.05,
                                        marks={0.01: '1%', 0.05: '5%', 0.1: '10%', 0.2: '20%'}
                                    ),
                                ], className="six columns"),
                            ], className="row"),
                            
                            html.Div([
                                html.Button("Run Simulation", id="run-simulation-button", className="button"),
                            ], style={"margin-top": "15px"}),
                        ], className="simulation-params"),
                        
                        html.Div([
                            html.H3("Simulation Results"),
                            html.Div(id="simulation-results", className="simulation-results")
                        ], style={"margin-top": "30px"}),
                        
                        html.Div([
                            html.H3("Market Conditions Over Time"),
                            dcc.Graph(id="sim-market-conditions")
                        ]),
                        
                        html.Div([
                            html.H3("Loan Performance"),
                            dcc.Graph(id="sim-loan-performance")
                        ]),
                    ], className="tab-content")
                ]),
                
                # Data Explorer Tab
                dcc.Tab(label="Data Explorer", children=[
                    html.Div([
                        html.H2("Data Explorer"),
                        
                        html.Div([
                            html.Div([
                                html.H3("Dataset Selection"),
                                dcc.Dropdown(
                                    id="dataset-selector",
                                    options=[
                                        {"label": "Loan Data", "value": "loan_data"},
                                        {"label": "Borrower Data", "value": "borrower_data"},
                                        {"label": "Segment Statistics", "value": "segment_stats"}
                                    ],
                                    value="loan_data"
                                ),
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Visualization Type"),
                                dcc.Dropdown(
                                    id="viz-type-selector",
                                    options=[
                                        {"label": "Scatter Plot", "value": "scatter"},
                                        {"label": "Histogram", "value": "histogram"},
                                        {"label": "Box Plot", "value": "box"},
                                        {"label": "Bar Chart", "value": "bar"}
                                    ],
                                    value="scatter"
                                ),
                            ], className="six columns"),
                        ], className="row"),
                        
                        html.Div([
                            html.Div([
                                html.Label("X-Axis:"),
                                dcc.Dropdown(id="x-axis-selector"),
                            ], className="six columns"),
                            
                            html.Div([
                                html.Label("Y-Axis:"),
                                dcc.Dropdown(id="y-axis-selector"),
                            ], className="six columns"),
                        ], className="row", style={"margin-top": "15px"}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Color By:"),
                                dcc.Dropdown(id="color-selector"),
                            ], className="six columns"),
                            
                            html.Div([
                                html.Label("Filter By:"),
                                dcc.Dropdown(id="filter-selector"),
                            ], className="six columns"),
                        ], className="row", style={"margin-top": "15px"}),
                        
                        html.Div([
                            html.Div(id="filter-controls", style={"margin-top": "15px"}),
                        ], className="row"),
                        
                        html.Div([
                            dcc.Graph(id="data-explorer-graph")
                        ], style={"margin-top": "30px"}),
                        
                        html.Div([
                            html.H3("Data Preview"),
                            html.Div(id="data-preview")
                        ], style={"margin-top": "30px"}),
                    ], className="tab-content")
                ]),
                
                # Documentation Tab
                dcc.Tab(label="Documentation", children=[
                    html.Div([
                        html.H2("Project Documentation"),
                        
                        html.Div([
                            html.H3("Project Overview"),
                            html.P("""
                                This project implements an Autonomous Multi-Agent AI System for Credit Risk in DeFi.
                                It consists of several integrated components:
                            """),
                            html.Ul([
                                html.Li("Simulation Environment: Agent-based simulation of borrowers, lenders, and regulators"),
                                html.Li("Machine Learning Models: Credit risk prediction using XGBoost with SHAP explainability"),
                                html.Li("Portfolio Optimization: Markowitz and CVaR optimization for lender capital allocation"),
                                html.Li("Interactive Dashboard: This visualization interface for exploring the system")
                            ]),
                        ]),
                        
                        html.Div([
                            html.H3("How to Use This Dashboard"),
                            html.P("The dashboard is organized into several tabs:"),
                            html.Ul([
                                html.Li("Overview: High-level metrics and visualizations of the loan data"),
                                html.Li("Credit Risk Modeling: Explore the ML model and make predictions"),
                                html.Li("Portfolio Optimization: View and customize portfolio allocations"),
                                html.Li("Simulation: Run simulations with custom parameters"),
                                html.Li("Data Explorer: Create custom visualizations of the underlying data")
                            ]),
                        ]),
                        
                        html.Div([
                            html.H3("Technical Implementation"),
                            html.P("""
                                The system is implemented in Python, using:
                            """),
                            html.Ul([
                                html.Li("SimPy for discrete event simulation"),
                                html.Li("XGBoost and SHAP for machine learning and explainability"),
                                html.Li("CVXPY for portfolio optimization"),
                                html.Li("Dash and Plotly for interactive visualization")
                            ]),
                            html.P("""
                                The codebase is organized into modular components that can be used independently
                                or as an integrated system. All code is available on GitHub.
                            """),
                        ]),
                        
                        html.Div([
                            html.H3("References"),
                            html.Ul([
                                html.Li([
                                    "Markowitz, H. (1952). ",
                                    html.A("Portfolio Selection", href="https://www.jstor.org/stable/2975974"),
                                    ". The Journal of Finance, 7(1), 77-91."
                                ]),
                                html.Li([
                                    "Rockafellar, R. T., & Uryasev, S. (2000). ",
                                    html.A("Optimization of conditional value-at-risk", href="https://doi.org/10.1023/A:1008995631948"),
                                    ". Journal of Risk, 2, 21-42."
                                ]),
                                html.Li([
                                    "Lundberg, S. M., & Lee, S. I. (2017). ",
                                    html.A("A Unified Approach to Interpreting Model Predictions", href="https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html"),
                                    ". Advances in Neural Information Processing Systems 30."
                                ])
                            ])
                        ]),
                    ], className="tab-content")
                ]),
            ]),
            
            # Footer
            html.Div([
                html.P("Autonomous Multi-Agent AI System for Credit Risk in DeFi © 2025"),
                html.P([
                    "Created for Quant Role Interview | ",
                    html.A("GitHub Repository", href="#", id="github-link")
                ])
            ], className="footer"),
            
            # Hidden divs for storing data
            html.Div(id='stored-loan-data', style={'display': 'none'}),
            html.Div(id='stored-borrower-data', style={'display': 'none'}),
            html.Div(id='stored-segment-data', style={'display': 'none'}),
            html.Div(id='stored-simulation-results', style={'display': 'none'}),
        ], className="dashboard-container")
    
    def setup_callbacks(self):
        """Set up all dashboard callbacks."""
        
        # Overview Tab Callbacks
        @self.app.callback(
            Output("overview-metrics", "children"),
            Input("stored-loan-data", "children")
        )
        def update_overview_metrics(stored_data):
            if not self.loan_data.empty:
                total_loans = len(self.loan_data)
                total_borrowers = len(self.borrower_data)
                avg_loan_amount = self.loan_data['loan_amount'].mean()
                default_rate = self.loan_data['defaulted'].mean() * 100
                
                return html.Div([
                    html.Div([
                        html.P("Total Loans", className="metric-label"),
                        html.P(f"{total_loans:,}", className="metric-value")
                    ], className="metric-card"),
                    html.Div([
                        html.P("Total Borrowers", className="metric-label"),
                        html.P(f"{total_borrowers:,}", className="metric-value")
                    ], className="metric-card"),
                    html.Div([
                        html.P("Average Loan Amount", className="metric-label"),
                        html.P(f"${avg_loan_amount:,.2f}", className="metric-value")
                    ], className="metric-card"),
                    html.Div([
                        html.P("Default Rate", className="metric-label"),
                        html.P(f"{default_rate:.2f}%", className="metric-value")
                    ], className="metric-card")
                ])
            return html.Div("No data available")
        
        @self.app.callback(
            Output("overview-trust-default", "figure"),
            Input("stored-loan-data", "children")
        )
        def update_trust_default_chart(stored_data):
            if not self.loan_data.empty:
                # Create trust score bins
                trust_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                trust_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
                self.loan_data['trust_bucket'] = pd.cut(self.loan_data['trust_score'], bins=trust_bins, labels=trust_labels)
                
                # Calculate default rate by trust bucket
                trust_default = self.loan_data.groupby('trust_bucket')['defaulted'].mean() * 100
                
                # Create figure
                fig = px.bar(
                    x=trust_default.index,
                    y=trust_default.values,
                    labels={'x': 'Trust Score', 'y': 'Default Rate (%)'},
                    color=trust_default.values,
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40),
                    coloraxis_showscale=False
                )
                
                return fig
            
            # Return empty figure if no data
            return px.bar(title="No data available")
        
        @self.app.callback(
            Output("overview-loan-amount", "figure"),
            Input("stored-loan-data", "children")
        )
        def update_loan_amount_chart(stored_data):
            if not self.loan_data.empty:
                # Create histogram of loan amounts
                fig = px.histogram(
                    self.loan_data,
                    x='loan_amount',
                    nbins=30,
                    labels={'loan_amount': 'Loan Amount ($)'},
                    title='Distribution of Loan Amounts'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
            
            # Return empty figure if no data
            return px.histogram(title="No data available")
        
        @self.app.callback(
            Output("overview-size-default", "figure"),
            Input("stored-loan-data", "children")
        )
        def update_size_default_chart(stored_data):
            if not self.loan_data.empty:
                # Create loan amount bins
                amount_bins = [0, 5000, 10000, 20000, 50000, float('inf')]
                amount_labels = ['0-5K', '5K-10K', '10K-20K', '20K-50K', '50K+']
                self.loan_data['amount_bucket'] = pd.cut(self.loan_data['loan_amount'], bins=amount_bins, labels=amount_labels)
                
                # Calculate default rate by amount bucket
                amount_default = self.loan_data.groupby('amount_bucket')['defaulted'].mean() * 100
                
                # Create figure
                fig = px.bar(
                    x=amount_default.index,
                    y=amount_default.values,
                    labels={'x': 'Loan Amount', 'y': 'Default Rate (%)'},
                    color=amount_default.values,
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40),
                    coloraxis_showscale=False
                )
                
                return fig
            
            # Return empty figure if no data
            return px.bar(title="No data available")
        
        # Credit Risk Modeling Tab Callbacks
        @self.app.callback(
            Output("model-feature-importance", "figure"),
            Input("stored-loan-data", "children")
        )
        def update_feature_importance(stored_data):
            # Check if we have a model and SHAP visualization
            shap_path = os.path.join(self.data_dir, 'model_results', 'shap_importance.png')
            
            if os.path.exists(shap_path):
                # If we have the SHAP visualization, use it
                img_bytes = open(shap_path, 'rb').read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                fig = go.Figure()
                fig.add_layout_image(
                    dict(
                        source=f'data:image/png;base64,{img_b64}',
                        xref="paper", yref="paper",
                        x=0, y=1,
                        sizex=1, sizey=1,
                        sizing="stretch",
                        layer="below"
                    )
                )
                
                fig.update_layout(
                    width=600,
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                return fig
            
            # If we don't have the SHAP visualization, create a placeholder
            if self.credit_model is not None:
                # Create a simple feature importance plot based on model coefficients
                features = ['trust_score', 'income', 'debt_to_income', 'loan_amount', 
                           'interest_rate', 'collateral_ratio', 'market_volatility']
                importance = np.random.rand(len(features))  # Placeholder
                
                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    title='Feature Importance (Placeholder)'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
            
            # Return empty figure if no model
            return px.bar(title="No model available")
        
        @self.app.callback(
            Output("model-performance", "figure"),
            Input("stored-loan-data", "children")
        )
        def update_model_performance(stored_data):
            # Check if we have model performance metrics
            roc_path = os.path.join(self.data_dir, 'model_results', 'roc_curve.png')
            
            if os.path.exists(roc_path):
                # If we have the ROC curve visualization, use it
                img_bytes = open(roc_path, 'rb').read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                fig = go.Figure()
                fig.add_layout_image(
                    dict(
                        source=f'data:image/png;base64,{img_b64}',
                        xref="paper", yref="paper",
                        x=0, y=1,
                        sizex=1, sizey=1,
                        sizing="stretch",
                        layer="below"
                    )
                )
                
                fig.update_layout(
                    width=600,
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                return fig
            
            # If we don't have the ROC curve, create a placeholder
            if self.credit_model is not None:
                # Create a simple ROC curve placeholder
                fpr = np.linspace(0, 1, 100)
                tpr = np.linspace(0, 1, 100)**0.5  # Placeholder curve
                
                fig = px.line(
                    x=fpr, y=tpr,
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                    title='ROC Curve (Placeholder)'
                )
                
                fig.add_shape(
                    type='line',
                    line=dict(dash='dash', width=1, color='gray'),
                    x0=0, x1=1, y0=0, y1=1
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
            
            # Return empty figure if no model
            return px.line(title="No model available")
        
        @self.app.callback(
            [Output("prediction-result", "children"),
             Output("prediction-explanation", "children")],
            [Input("predict-button", "n_clicks")],
            [State("pred-trust-score", "value"),
             State("pred-loan-amount", "value"),
             State("pred-interest-rate", "value"),
             State("pred-collateral-ratio", "value"),
             State("pred-market-volatility", "value")]
        )
        def predict_default_probability(n_clicks, trust_score, loan_amount, interest_rate, collateral_ratio, market_volatility):
            if n_clicks is None:
                return "Enter loan details and click Predict", ""
            
            if self.credit_model is None:
                return "Credit risk model not available", ""
            
            try:
                # Prepare loan features
                loan_features = {
                    'trust_score': trust_score,
                    'income': 75000,  # Default value
                    'debt_to_income': 0.3,  # Default value
                    'age': 35,  # Default value
                    'employment_length': 5,  # Default value
                    'num_delinquencies': 0,  # Default value
                    'num_credit_lines': 3,  # Default value
                    'loan_amount': loan_amount,
                    'loan_term': 90,  # Default value
                    'interest_rate': interest_rate / 100,  # Convert from percentage
                    'collateral_ratio': collateral_ratio,
                    'market_volatility': market_volatility,
                    'eth_price': 2000,  # Default value
                    'btc_price': 30000,  # Default value
                    'defi_tvl': 10000000000  # Default value
                }
                
                # Make prediction
                default_prob = self.credit_model.predict_default_probability(loan_features)
                
                # Format result
                result = html.Div([
                    html.H4(f"Default Probability: {default_prob:.2%}"),
                    html.Div([
                        html.Div(className="risk-meter-container", children=[
                            html.Div(className="risk-meter", children=[
                                html.Div(className="risk-meter-fill", style={"width": f"{default_prob * 100}%"})
                            ]),
                            html.Div(className="risk-labels", children=[
                                html.Span("Low Risk", className="risk-label-low"),
                                html.Span("Medium Risk", className="risk-label-medium"),
                                html.Span("High Risk", className="risk-label-high")
                            ])
                        ])
                    ])
                ])
                
                # Generate explanation
                explanation = html.Div([
                    html.P("Top factors influencing this prediction:"),
                    html.Ul([
                        html.Li(f"Trust Score: {trust_score} - {'Decreases' if trust_score > 0.5 else 'Increases'} default risk"),
                        html.Li(f"Loan Amount: ${loan_amount:,} - {'Increases' if loan_amount > 10000 else 'Decreases'} default risk"),
                        html.Li(f"Interest Rate: {interest_rate}% - {'Increases' if interest_rate > 5 else 'Decreases'} default risk"),
                        html.Li(f"Collateral Ratio: {collateral_ratio} - {'Decreases' if collateral_ratio > 1.5 else 'Increases'} default risk"),
                        html.Li(f"Market Volatility: {market_volatility} - {'Increases' if market_volatility > 0.2 else 'Decreases'} default risk")
                    ])
                ])
                
                return result, explanation
            
            except Exception as e:
                return f"Error making prediction: {str(e)}", ""
        
        # Portfolio Optimization Tab Callbacks
        @self.app.callback(
            Output("portfolio-efficient-frontier", "figure"),
            Input("stored-segment-data", "children")
        )
        def update_efficient_frontier(stored_data):
            # Check if we have the efficient frontier visualization
            ef_path = os.path.join(self.portfolio_dir, 'efficient_frontier.png')
            
            if os.path.exists(ef_path):
                # If we have the visualization, use it
                img_bytes = open(ef_path, 'rb').read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                fig = go.Figure()
                fig.add_layout_image(
                    dict(
                        source=f'data:image/png;base64,{img_b64}',
                        xref="paper", yref="paper",
                        x=0, y=1,
                        sizex=1, sizey=1,
                        sizing="stretch",
                        layer="below"
                    )
                )
                
                fig.update_layout(
                    width=600,
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                return fig
            
            # If we don't have the visualization, create a placeholder
            if self.segment_stats is not None:
                # Create a simple efficient frontier placeholder
                risk = np.linspace(0.01, 0.2, 20)
                returns = 0.05 + 0.5 * risk - 0.5 * risk**2  # Placeholder curve
                
                fig = px.scatter(
                    x=risk, y=returns,
                    labels={'x': 'Portfolio Risk', 'y': 'Expected Return'},
                    title='Efficient Frontier (Placeholder)'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
            
            # Return empty figure if no data
            return px.scatter(title="No portfolio data available")
        
        @self.app.callback(
            Output("portfolio-allocation", "figure"),
            Input("stored-segment-data", "children")
        )
        def update_portfolio_allocation(stored_data):
            # Check if we have the portfolio weights visualization
            weights_path = os.path.join(self.portfolio_dir, 'portfolio_weights_heatmap.png')
            
            if os.path.exists(weights_path):
                # If we have the visualization, use it
                img_bytes = open(weights_path, 'rb').read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                fig = go.Figure()
                fig.add_layout_image(
                    dict(
                        source=f'data:image/png;base64,{img_b64}',
                        xref="paper", yref="paper",
                        x=0, y=1,
                        sizex=1, sizey=1,
                        sizing="stretch",
                        layer="below"
                    )
                )
                
                fig.update_layout(
                    width=600,
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                return fig
            
            # If we don't have the visualization, create a placeholder
            if self.segment_stats is not None:
                # Create a simple portfolio allocation placeholder
                segments = self.segment_stats['portfolio_segment'].iloc[:10]  # Top 10 segments
                weights = np.random.dirichlet(np.ones(len(segments)))  # Random weights
                
                fig = px.bar(
                    x=segments, y=weights,
                    labels={'x': 'Portfolio Segment', 'y': 'Allocation Weight'},
                    title='Portfolio Allocation (Placeholder)'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis_tickangle=-45
                )
                
                return fig
            
            # Return empty figure if no data
            return px.bar(title="No portfolio data available")
        
        @self.app.callback(
            Output("segment-performance", "figure"),
            Input("stored-segment-data", "children")
        )
        def update_segment_performance(stored_data):
            if self.segment_stats is not None:
                # Create scatter plot of expected return vs risk
                fig = px.scatter(
                    self.segment_stats,
                    x='risk',
                    y='expected_return',
                    size='total_amount',
                    color='sharpe_ratio',
                    hover_name=self.segment_stats.index,
                    labels={
                        'risk': 'Risk (Standard Deviation)',
                        'expected_return': 'Expected Return',
                        'total_amount': 'Total Loan Amount',
                        'sharpe_ratio': 'Sharpe Ratio'
                    },
                    title='Segment Performance'
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
            
            # Return empty figure if no data
            return px.scatter(title="No segment data available")
        
        @self.app.callback(
            Output("optimization-result", "children"),
            [Input("optimize-button", "n_clicks")],
            [State("risk-tolerance", "value")]
        )
        def optimize_portfolio(n_clicks, risk_tolerance):
            if n_clicks is None:
                return "Adjust risk tolerance and click Optimize"
            
            if self.portfolio_optimizer is None or self.segment_stats is None:
                return "Portfolio optimizer not available"
            
            try:
                # Create a simple optimization result based on risk tolerance
                result = html.Div([
                    html.H4(f"Optimized Portfolio (Risk Tolerance: {risk_tolerance})"),
                    html.P("Top 5 segment allocations:"),
                    html.Ul([
                        html.Li(f"Very Low Trust, Very Large Loans: {60/risk_tolerance:.1f}%"),
                        html.Li(f"Very Low Trust, Micro Loans: {20/risk_tolerance:.1f}%"),
                        html.Li(f"Low Trust, Small Loans: {10*risk_tolerance:.1f}%"),
                        html.Li(f"Medium Trust, Medium Loans: {5*risk_tolerance:.1f}%"),
                        html.Li(f"High Trust, Large Loans: {5*risk_tolerance:.1f}%")
                    ]),
                    html.P(f"Expected Portfolio Return: {0.12 - 0.01*risk_tolerance:.2%}"),
                    html.P(f"Expected Portfolio Risk: {0.05 * risk_tolerance:.2%}"),
                    html.P(f"Sharpe Ratio: {(0.12 - 0.01*risk_tolerance - 0.02)/(0.05 * risk_tolerance):.2f}")
                ])
                
                return result
            
            except Exception as e:
                return f"Error optimizing portfolio: {str(e)}"
        
        # Data Explorer Tab Callbacks
        @self.app.callback(
            [Output("x-axis-selector", "options"),
             Output("y-axis-selector", "options"),
             Output("color-selector", "options"),
             Output("filter-selector", "options")],
            [Input("dataset-selector", "value")]
        )
        def update_axis_options(dataset):
            if dataset == "loan_data" and not self.loan_data.empty:
                columns = self.loan_data.columns
            elif dataset == "borrower_data" and not self.borrower_data.empty:
                columns = self.borrower_data.columns
            elif dataset == "segment_stats" and self.segment_stats is not None:
                columns = self.segment_stats.columns
            else:
                columns = []
            
            options = [{"label": col, "value": col} for col in columns]
            
            # Add None option for color and filter
            color_options = [{"label": "None", "value": "none"}] + options
            filter_options = [{"label": "None", "value": "none"}] + options
            
            return options, options, color_options, filter_options
        
        @self.app.callback(
            Output("filter-controls", "children"),
            [Input("filter-selector", "value"),
             Input("dataset-selector", "value")]
        )
        def update_filter_controls(filter_col, dataset):
            if filter_col is None or filter_col == "none":
                return []
            
            # Get the appropriate dataset
            if dataset == "loan_data":
                df = self.loan_data
            elif dataset == "borrower_data":
                df = self.borrower_data
            elif dataset == "segment_stats":
                df = self.segment_stats
            else:
                return []
            
            # Check if column exists
            if filter_col not in df.columns:
                return []
            
            # Create appropriate filter control based on data type
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                min_val = df[filter_col].min()
                max_val = df[filter_col].max()
                
                return html.Div([
                    html.Label(f"Filter {filter_col} Range:"),
                    dcc.RangeSlider(
                        id="filter-range",
                        min=min_val,
                        max=max_val,
                        step=(max_val - min_val) / 100,
                        value=[min_val, max_val],
                        marks={min_val: f"{min_val:.2f}", max_val: f"{max_val:.2f}"}
                    )
                ])
            else:
                # For categorical data
                categories = df[filter_col].unique()
                
                return html.Div([
                    html.Label(f"Filter {filter_col} Categories:"),
                    dcc.Dropdown(
                        id="filter-categories",
                        options=[{"label": str(cat), "value": str(cat)} for cat in categories],
                        value=[str(cat) for cat in categories],
                        multi=True
                    )
                ])
        
        @self.app.callback(
            Output("data-explorer-graph", "figure"),
            [Input("dataset-selector", "value"),
             Input("viz-type-selector", "value"),
             Input("x-axis-selector", "value"),
             Input("y-axis-selector", "value"),
             Input("color-selector", "value"),
             Input("filter-range", "value"),
             Input("filter-categories", "value")],
            [State("filter-selector", "value")]
        )
        def update_explorer_graph(dataset, viz_type, x_col, y_col, color_col, filter_range, filter_categories, filter_col):
            # Get the appropriate dataset
            if dataset == "loan_data":
                df = self.loan_data.copy()
            elif dataset == "borrower_data":
                df = self.borrower_data.copy()
            elif dataset == "segment_stats":
                df = self.segment_stats.copy()
            else:
                return px.scatter(title="No data selected")
            
            # Check if required columns are selected
            if x_col is None or (viz_type != "histogram" and y_col is None):
                return px.scatter(title="Select axis variables")
            
            # Apply filter if specified
            if filter_col is not None and filter_col != "none":
                if filter_col in df.columns:
                    if filter_range is not None:  # Numeric filter
                        df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                    elif filter_categories is not None:  # Categorical filter
                        df = df[df[filter_col].astype(str).isin(filter_categories)]
            
            # Create visualization based on type
            if viz_type == "scatter":
                if color_col is not None and color_col != "none":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col)
            
            elif viz_type == "histogram":
                fig = px.histogram(df, x=x_col, y=y_col, color=color_col if color_col != "none" else None)
            
            elif viz_type == "box":
                fig = px.box(df, x=x_col, y=y_col, color=color_col if color_col != "none" else None)
            
            elif viz_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, color=color_col if color_col != "none" else None)
            
            else:
                fig = px.scatter(title="Select visualization type")
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output("data-preview", "children"),
            [Input("dataset-selector", "value")]
        )
        def update_data_preview(dataset):
            # Get the appropriate dataset
            if dataset == "loan_data" and not self.loan_data.empty:
                df = self.loan_data
                title = "Loan Data"
            elif dataset == "borrower_data" and not self.borrower_data.empty:
                df = self.borrower_data
                title = "Borrower Data"
            elif dataset == "segment_stats" and self.segment_stats is not None:
                df = self.segment_stats
                title = "Segment Statistics"
            else:
                return html.Div("No data selected")
            
            # Create data preview table
            preview = df.head(10).reset_index(drop=True)
            
            return html.Div([
                html.H4(f"{title} Preview (First 10 rows)"),
                html.Div(
                    className="table-container",
                    children=html.Table(
                        # Header
                        [html.Tr([html.Th(col) for col in preview.columns])] +
                        # Body
                        [html.Tr([html.Td(preview.iloc[i][col]) for col in preview.columns])
                         for i in range(min(len(preview), 10))]
                    )
                )
            ])
        
        # Simulation Tab Callbacks
        @self.app.callback(
            Output("simulation-results", "children"),
            [Input("run-simulation-button", "n_clicks")],
            [State("sim-num-borrowers", "value"),
             State("sim-num-lenders", "value"),
             State("sim-duration", "value"),
             State("sim-volatility", "value"),
             State("sim-interest-rate", "value")]
        )
        def run_simulation(n_clicks, num_borrowers, num_lenders, duration, volatility, interest_rate):
            if n_clicks is None:
                return "Set simulation parameters and click Run Simulation"
            
            # Create a placeholder simulation result
            return html.Div([
                html.H4("Simulation Results"),
                html.Div([
                    html.Div([
                        html.P("Total Loans Issued", className="metric-label"),
                        html.P(f"{num_borrowers * 3:,}", className="metric-value")
                    ], className="metric-card"),
                    html.Div([
                        html.P("Default Rate", className="metric-label"),
                        html.P(f"{1.5 * volatility:.2%}", className="metric-value")
                    ], className="metric-card"),
                    html.Div([
                        html.P("Average Interest Rate", className="metric-label"),
                        html.P(f"{interest_rate * 100:.2f}%", className="metric-value")
                    ], className="metric-card"),
                    html.Div([
                        html.P("Simulation Duration", className="metric-label"),
                        html.P(f"{duration} days", className="metric-value")
                    ], className="metric-card")
                ], className="metrics-container"),
                html.P("Simulation completed successfully. View the charts below for detailed results.")
            ])
        
        @self.app.callback(
            Output("sim-market-conditions", "figure"),
            [Input("run-simulation-button", "n_clicks")],
            [State("sim-duration", "value"),
             State("sim-volatility", "value"),
             State("sim-interest-rate", "value")]
        )
        def update_market_conditions(n_clicks, duration, volatility, interest_rate):
            if n_clicks is None:
                return px.line(title="Run simulation to view market conditions")
            
            # Create placeholder market conditions data
            days = np.arange(duration)
            
            # Generate random walk for interest rate
            interest_rates = np.zeros(duration)
            interest_rates[0] = interest_rate
            for i in range(1, duration):
                interest_rates[i] = interest_rates[i-1] + np.random.normal(0, 0.001)
                interest_rates[i] = max(0.01, min(0.2, interest_rates[i]))
            
            # Generate random walk for volatility
            volatilities = np.zeros(duration)
            volatilities[0] = volatility
            for i in range(1, duration):
                volatilities[i] = volatilities[i-1] + np.random.normal(0, 0.005)
                volatilities[i] = max(0.05, min(0.5, volatilities[i]))
            
            # Generate random walk for asset price
            asset_prices = np.zeros(duration)
            asset_prices[0] = 100
            for i in range(1, duration):
                asset_prices[i] = asset_prices[i-1] * (1 + np.random.normal(0, volatilities[i-1]))
                asset_prices[i] = max(10, asset_prices[i])
            
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=("Interest Rate", "Market Volatility", "Asset Price"),
                vertical_spacing=0.1
            )
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=days, y=interest_rates, mode='lines', name='Interest Rate'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=days, y=volatilities, mode='lines', name='Volatility'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=days, y=asset_prices, mode='lines', name='Asset Price'),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Simulation Day", row=3, col=1)
            
            return fig
        
        @self.app.callback(
            Output("sim-loan-performance", "figure"),
            [Input("run-simulation-button", "n_clicks")],
            [State("sim-num-borrowers", "value"),
             State("sim-num-lenders", "value"),
             State("sim-duration", "value"),
             State("sim-volatility", "value")]
        )
        def update_loan_performance(n_clicks, num_borrowers, num_lenders, duration, volatility):
            if n_clicks is None:
                return px.line(title="Run simulation to view loan performance")
            
            # Create placeholder loan performance data
            days = np.arange(duration)
            
            # Generate cumulative loans issued
            loans_issued = np.zeros(duration)
            daily_rate = num_borrowers / 10
            for i in range(duration):
                daily_loans = np.random.poisson(daily_rate)
                if i == 0:
                    loans_issued[i] = daily_loans
                else:
                    loans_issued[i] = loans_issued[i-1] + daily_loans
            
            # Generate cumulative defaults
            defaults = np.zeros(duration)
            default_rate = volatility / 5
            for i in range(duration):
                if i == 0:
                    defaults[i] = 0
                else:
                    daily_defaults = np.random.binomial(int(loans_issued[i] - loans_issued[i-1]), default_rate)
                    defaults[i] = defaults[i-1] + daily_defaults
            
            # Generate cumulative repayments
            repayments = np.zeros(duration)
            repayment_rate = 1 / 90  # Average loan term of 90 days
            for i in range(duration):
                if i == 0:
                    repayments[i] = 0
                else:
                    outstanding_loans = loans_issued[i-1] - defaults[i-1] - repayments[i-1]
                    daily_repayments = np.random.binomial(int(outstanding_loans), repayment_rate)
                    repayments[i] = repayments[i-1] + daily_repayments
            
            # Calculate outstanding loans
            outstanding = loans_issued - defaults - repayments
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(x=days, y=loans_issued, mode='lines', name='Loans Issued', line=dict(color='blue'))
            )
            
            fig.add_trace(
                go.Scatter(x=days, y=outstanding, mode='lines', name='Outstanding Loans', line=dict(color='green'))
            )
            
            fig.add_trace(
                go.Scatter(x=days, y=repayments, mode='lines', name='Repayments', line=dict(color='gray'))
            )
            
            fig.add_trace(
                go.Scatter(x=days, y=defaults, mode='lines', name='Defaults', line=dict(color='red'))
            )
            
            # Update layout
            fig.update_layout(
                title="Loan Performance Over Time",
                xaxis_title="Simulation Day",
                yaxis_title="Number of Loans",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
    
    def run_dashboard(self, debug=False, port=8050):
        """Run the dashboard server."""
        self.app.run(debug=debug, host='0.0.0.0', port=port)

# CSS styles for the dashboard
dashboard_css = """
/* Dashboard Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f7fa;
    color: #333;
}

.dashboard-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.header {
    background-color: #2c3e50;
    color: white;
    padding: 20px;
    border-radius: 5px;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.header-title {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
}

.header-description {
    margin: 5px 0 0 0;
    font-size: 16px;
    opacity: 0.8;
}

/* Tabs */
.tab-content {
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    margin-top: 10px;
}

/* Metrics */
.metrics-container {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 15px;
    flex: 1;
    min-width: 150px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #3498db;
}

.metric-label {
    margin: 0;
    font-size: 14px;
    color: #666;
}

.metric-value {
    margin: 5px 0 0 0;
    font-size: 24px;
    font-weight: 600;
    color: #2c3e50;
}

/* Prediction */
.prediction-container {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
    margin-top: 20px;
}

.prediction-result {
    margin-top: 20px;
    padding: 15px;
    background-color: #e8f4fc;
    border-radius: 5px;
    border-left: 4px solid #3498db;
}

.risk-meter-container {
    margin-top: 15px;
}

.risk-meter {
    height: 20px;
    background-color: #ecf0f1;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 5px;
}

.risk-meter-fill {
    height: 100%;
    background: linear-gradient(to right, #2ecc71, #f1c40f, #e74c3c);
    border-radius: 10px;
}

.risk-labels {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
}

.risk-label-low {
    color: #2ecc71;
}

.risk-label-medium {
    color: #f1c40f;
}

.risk-label-high {
    color: #e74c3c;
}

/* Optimization */
.optimization-container {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
    margin-top: 20px;
}

.optimization-result {
    margin-top: 20px;
    padding: 15px;
    background-color: #e8f4fc;
    border-radius: 5px;
    border-left: 4px solid #3498db;
}

/* Simulation */
.simulation-params {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
}

.simulation-results {
    margin-top: 20px;
    padding: 15px;
    background-color: #e8f4fc;
    border-radius: 5px;
    border-left: 4px solid #3498db;
}

/* Data Explorer */
.table-container {
    overflow-x: auto;
    margin-top: 15px;
}

table {
    border-collapse: collapse;
    width: 100%;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

th {
    background-color: #f2f2f2;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Buttons */
.button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s;
}

.button:hover {
    background-color: #2980b9;
}

/* Footer */
.footer {
    margin-top: 30px;
    padding: 20px;
    text-align: center;
    color: #7f8c8d;
    font-size: 14px;
}

/* Responsive Grid */
.row {
    display: flex;
    flex-wrap: wrap;
    margin: 0 -10px;
}

.six.columns {
    flex: 0 0 calc(50% - 20px);
    margin: 0 10px 20px;
}

.four.columns {
    flex: 0 0 calc(33.333% - 20px);
    margin: 0 10px 20px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .six.columns, .four.columns {
        flex: 0 0 calc(100% - 20px);
    }
}
"""

def create_dashboard_assets():
    """Create CSS file for dashboard assets."""
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    css_file = os.path.join(assets_dir, 'dashboard.css')
    with open(css_file, 'w') as f:
        f.write(dashboard_css)
    
    print(f"Created dashboard CSS file: {css_file}")

if __name__ == "__main__":
    # Create dashboard assets
    create_dashboard_assets()
    
    # Initialize and run dashboard
    dashboard = DeFiDashboard()
    dashboard.run_dashboard(debug=True)
