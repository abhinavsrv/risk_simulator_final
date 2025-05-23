# Risk Simulator - Enterprise Financial Risk Analytics Platform

Risk Simulator is a comprehensive analytics platform for financial risk assessment and portfolio optimization. This enterprise-grade solution combines advanced simulation, machine learning, and portfolio theory to provide actionable insights for financial institutions.

## Key Features

### Multi-Agent Simulation Environment
- Agent-based modeling of borrowers, lenders, and market regulators
- Dynamic market conditions with configurable volatility
- Realistic financial behavior modeling based on empirical data

### Machine Learning Risk Models
- Advanced credit risk prediction with XGBoost
- Model explainability with SHAP values
- Feature importance visualization and analysis
- Interactive risk assessment tools

### Portfolio Optimization
- Markowitz Mean-Variance Optimization
- Conditional Value at Risk (CVaR) modeling
- Efficient frontier visualization
- Risk-return tradeoff analysis
- Segment-based portfolio allocation

### Interactive Analytics Dashboard
- Real-time data visualization
- Customizable risk parameters
- Portfolio optimization tools
- Market simulation controls
- Data exploration capabilities

## Technology Stack

- **Simulation**: SimPy, NumPy, Pandas
- **Machine Learning**: XGBoost, Scikit-learn, SHAP
- **Portfolio Optimization**: CVXPY, SciPy
- **Visualization**: Plotly.js, D3.js
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy

## Project Structure

```
risk_simulator/
├── agents/                  # Agent models for simulation
├── data/                    # Data files and results
│   ├── model_results/       # ML model outputs
│   ├── portfolio_results/   # Portfolio optimization results
│   └── visualizations/      # Generated visualizations
├── docs/                    # Documentation and static site
├── ml_models/               # Machine learning models
│   ├── credit_risk_model.py # Credit risk prediction
│   ├── portfolio_optimizer.py # Portfolio optimization
│   └── synthetic_data_generator.py # Data generation
├── simulation_env/          # Simulation environment
└── dashboard/               # Interactive dashboard
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the simulation: `python simulation_env/simulation_env_integrated.py`
4. Generate ML models: `python ml_models/credit_risk_model.py`
5. Optimize portfolios: `python ml_models/portfolio_optimizer.py`
6. Launch the dashboard: `python dashboard/dashboard.py`

## Dashboard

The interactive dashboard provides a comprehensive interface for exploring risk models, optimizing portfolios, and running market simulations. Access the static version in the `docs` folder or run the dynamic version using the dashboard module.

## References

- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2, 21-42.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
