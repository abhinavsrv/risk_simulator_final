// DeFi Credit Risk Dashboard JavaScript

// Tab Navigation
function openTab(evt, tabName) {
    // Hide all tab content
    var tabContent = document.getElementsByClassName("tab-content");
    for (var i = 0; i < tabContent.length; i++) {
        tabContent[i].classList.remove("active");
    }

    // Remove active class from all tab buttons
    var tabButtons = document.getElementsByClassName("tab-button");
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }

    // Show the selected tab and mark its button as active
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Load data and initialize charts
    initializeOverviewCharts();
    initializeSimulationCharts();
    initializeDataExplorer();
    
    // Set up event listeners for range inputs
    document.getElementById('pred-trust-score').addEventListener('input', function() {
        document.getElementById('trust-score-value').textContent = this.value;
    });
    
    document.getElementById('pred-market-volatility').addEventListener('input', function() {
        document.getElementById('volatility-value').textContent = this.value;
    });
    
    document.getElementById('risk-tolerance').addEventListener('input', function() {
        document.getElementById('risk-tolerance-value').textContent = this.value;
    });
    
    document.getElementById('sim-volatility').addEventListener('input', function() {
        document.getElementById('sim-volatility-value').textContent = this.value;
    });
});

// Overview Tab Charts
function initializeOverviewCharts() {
    // Default Trust Score Default Rate Chart
    var trustLabels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'];
    var defaultRates = [18.5, 12.3, 7.8, 4.2, 1.5];
    
    var trustDefaultChart = Plotly.newPlot('overview-trust-default', [{
        x: trustLabels,
        y: defaultRates,
        type: 'bar',
        marker: {
            color: defaultRates,
            colorscale: 'RdYlGn_r'
        }
    }], {
        title: 'Default Rate by Trust Score',
        xaxis: { title: 'Trust Score' },
        yaxis: { title: 'Default Rate (%)' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    });
    
    // Loan Amount Distribution Chart
    var loanAmounts = [];
    for (var i = 0; i < 1000; i++) {
        loanAmounts.push(Math.exp(Math.random() * Math.log(100000)));
    }
    
    var loanAmountChart = Plotly.newPlot('overview-loan-amount', [{
        x: loanAmounts,
        type: 'histogram',
        nbinsx: 30,
        marker: {
            color: 'rgba(52, 152, 219, 0.7)',
            line: {
                color: 'rgba(52, 152, 219, 1)',
                width: 1
            }
        }
    }], {
        title: 'Distribution of Loan Amounts',
        xaxis: { title: 'Loan Amount ($)' },
        yaxis: { title: 'Count' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    });
    
    // Default Rate by Loan Size Chart
    var sizeLabels = ['0-5K', '5K-10K', '10K-20K', '20K-50K', '50K+'];
    var sizeDefaultRates = [5.2, 6.8, 8.3, 10.5, 15.7];
    
    var sizeDefaultChart = Plotly.newPlot('overview-size-default', [{
        x: sizeLabels,
        y: sizeDefaultRates,
        type: 'bar',
        marker: {
            color: sizeDefaultRates,
            colorscale: 'RdYlGn_r'
        }
    }], {
        title: 'Default Rate by Loan Size',
        xaxis: { title: 'Loan Amount' },
        yaxis: { title: 'Default Rate (%)' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    });
}

// Credit Risk Prediction
function predictDefault() {
    // Get input values
    var trustScore = parseFloat(document.getElementById('pred-trust-score').value);
    var loanAmount = parseFloat(document.getElementById('pred-loan-amount').value);
    var interestRate = parseFloat(document.getElementById('pred-interest-rate').value) / 100;
    var collateralRatio = parseFloat(document.getElementById('pred-collateral-ratio').value);
    var marketVolatility = parseFloat(document.getElementById('pred-market-volatility').value);
    
    // Simple model for demonstration (in a real app, this would call an API)
    var defaultProb = 0.5 - (trustScore * 0.4) + (loanAmount / 100000 * 0.1) + 
                     (interestRate * 0.5) - (collateralRatio * 0.1) + (marketVolatility * 0.5);
    
    // Ensure probability is between 0 and 1
    defaultProb = Math.max(0, Math.min(1, defaultProb));
    
    // Update the prediction result
    var resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = `
        <h4>Default Probability: ${(defaultProb * 100).toFixed(2)}%</h4>
        <div class="risk-meter-container">
            <div class="risk-meter">
                <div class="risk-meter-fill" style="width: ${defaultProb * 100}%"></div>
            </div>
            <div class="risk-labels">
                <span class="risk-label-low">Low Risk</span>
                <span class="risk-label-medium">Medium Risk</span>
                <span class="risk-label-high">High Risk</span>
            </div>
        </div>
    `;
    
    // Update the explanation
    var explanationDiv = document.getElementById('prediction-explanation');
    explanationDiv.innerHTML = `
        <p>The default probability is influenced by the following factors:</p>
        <ul>
            <li><strong>Trust Score:</strong> ${trustScore} (${trustScore < 0.5 ? 'Negative' : 'Positive'} impact)</li>
            <li><strong>Loan Amount:</strong> $${loanAmount.toLocaleString()} (${loanAmount > 20000 ? 'Negative' : 'Neutral'} impact)</li>
            <li><strong>Interest Rate:</strong> ${(interestRate * 100).toFixed(2)}% (${interestRate > 0.08 ? 'Negative' : 'Neutral'} impact)</li>
            <li><strong>Collateral Ratio:</strong> ${collateralRatio.toFixed(1)} (${collateralRatio < 1.5 ? 'Negative' : 'Positive'} impact)</li>
            <li><strong>Market Volatility:</strong> ${marketVolatility.toFixed(2)} (${marketVolatility > 0.3 ? 'Negative' : 'Neutral'} impact)</li>
        </ul>
        <p>The most significant factors in this prediction are trust score and market volatility.</p>
    `;
}

// Portfolio Optimization
function optimizePortfolio() {
    // Get risk tolerance
    var riskTolerance = parseFloat(document.getElementById('risk-tolerance').value);
    
    // Update the optimization result
    var resultDiv = document.getElementById('optimization-result');
    resultDiv.innerHTML = `
        <h4>Optimized Portfolio (Risk Tolerance: ${riskTolerance.toFixed(1)})</h4>
        <p>Top 5 segment allocations:</p>
        <ul>
            <li>Very Low Trust, Very Large Loans: ${(60/riskTolerance).toFixed(1)}%</li>
            <li>Very Low Trust, Micro Loans: ${(20/riskTolerance).toFixed(1)}%</li>
            <li>Low Trust, Small Loans: ${(10*riskTolerance).toFixed(1)}%</li>
            <li>Medium Trust, Medium Loans: ${(5*riskTolerance).toFixed(1)}%</li>
            <li>High Trust, Large Loans: ${(5*riskTolerance).toFixed(1)}%</li>
        </ul>
        <p>Expected Portfolio Return: ${(0.12 - 0.01*riskTolerance).toFixed(2)}%</p>
        <p>Expected Portfolio Risk: ${(0.05 * riskTolerance).toFixed(2)}%</p>
        <p>Sharpe Ratio: ${((0.12 - 0.01*riskTolerance - 0.02)/(0.05 * riskTolerance)).toFixed(2)}</p>
    `;
}

// Simulation
function initializeSimulationCharts() {
    // Empty placeholder charts
    Plotly.newPlot('sim-market-conditions', [{
        x: [0],
        y: [0],
        type: 'scatter',
        mode: 'lines',
        name: 'Run simulation to view data'
    }], {
        title: 'Market Conditions',
        xaxis: { title: 'Simulation Day' },
        yaxis: { title: 'Value' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    });
    
    Plotly.newPlot('sim-loan-performance', [{
        x: [0],
        y: [0],
        type: 'scatter',
        mode: 'lines',
        name: 'Run simulation to view data'
    }], {
        title: 'Loan Performance',
        xaxis: { title: 'Simulation Day' },
        yaxis: { title: 'Value' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    });
}

function runSimulation() {
    // Get simulation parameters
    var numBorrowers = parseInt(document.getElementById('sim-num-borrowers').value);
    var numLenders = parseInt(document.getElementById('sim-num-lenders').value);
    var duration = parseInt(document.getElementById('sim-duration').value);
    var volatility = parseFloat(document.getElementById('sim-volatility').value);
    var interestRate = parseFloat(document.getElementById('sim-interest-rate').value);
    
    // Update simulation results
    var resultsDiv = document.getElementById('simulation-results');
    resultsDiv.innerHTML = `
        <h4>Simulation Results</h4>
        <div class="metrics-container">
            <div class="metric-card">
                <p class="metric-label">Total Loans Issued</p>
                <p class="metric-value">${(numBorrowers * 3).toLocaleString()}</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">Default Rate</p>
                <p class="metric-value">${(1.5 * volatility * 100).toFixed(2)}%</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">Average Interest Rate</p>
                <p class="metric-value">${interestRate.toFixed(2)}%</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">Simulation Duration</p>
                <p class="metric-value">${duration} days</p>
            </div>
        </div>
        <p>Simulation completed successfully. View the charts below for detailed results.</p>
    `;
    
    // Generate market conditions data
    var days = Array.from({length: duration}, (_, i) => i);
    
    // Interest rates (random walk)
    var interestRates = [interestRate/100];
    for (var i = 1; i < duration; i++) {
        interestRates.push(interestRates[i-1] + (Math.random() - 0.5) * 0.002);
        interestRates[i] = Math.max(0.01, Math.min(0.2, interestRates[i]));
    }
    
    // Volatility (random walk)
    var volatilities = [volatility];
    for (var i = 1; i < duration; i++) {
        volatilities.push(volatilities[i-1] + (Math.random() - 0.5) * 0.01);
        volatilities[i] = Math.max(0.05, Math.min(0.5, volatilities[i]));
    }
    
    // Asset price (random walk with volatility)
    var assetPrices = [100];
    for (var i = 1; i < duration; i++) {
        assetPrices.push(assetPrices[i-1] * (1 + (Math.random() - 0.5) * volatilities[i-1]));
        assetPrices[i] = Math.max(10, assetPrices[i]);
    }
    
    // Update market conditions chart
    Plotly.newPlot('sim-market-conditions', [
        {
            x: days,
            y: interestRates.map(r => r * 100), // Convert to percentage
            type: 'scatter',
            mode: 'lines',
            name: 'Interest Rate (%)'
        },
        {
            x: days,
            y: volatilities,
            type: 'scatter',
            mode: 'lines',
            name: 'Volatility'
        },
        {
            x: days,
            y: assetPrices,
            type: 'scatter',
            mode: 'lines',
            name: 'Asset Price'
        }
    ], {
        title: 'Market Conditions',
        xaxis: { title: 'Simulation Day' },
        yaxis: { title: 'Value' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        legend: { orientation: 'h', y: 1.1 }
    });
    
    // Generate loan performance data
    var loansIssued = Array(duration).fill(0);
    var loansActive = Array(duration).fill(0);
    var loansDefaulted = Array(duration).fill(0);
    var loansRepaid = Array(duration).fill(0);
    
    // Simple model for demonstration
    for (var i = 0; i < duration; i++) {
        // New loans issued each day (random with trend)
        loansIssued[i] = i === 0 ? 0 : loansIssued[i-1] + Math.floor(numBorrowers * 0.03 * (1 + Math.random() * 0.5));
        
        // Active loans (cumulative issued minus repaid and defaulted)
        loansActive[i] = i === 0 ? 0 : loansIssued[i] - loansRepaid[i] - loansDefaulted[i];
        
        // Defaults (based on volatility)
        loansDefaulted[i] = i === 0 ? 0 : Math.floor(loansActive[i-1] * volatilities[i] * 0.05);
        
        // Repayments (based on duration)
        loansRepaid[i] = i === 0 ? 0 : Math.floor(loansActive[i-1] * (1/duration) * 10);
    }
    
    // Update loan performance chart
    Plotly.newPlot('sim-loan-performance', [
        {
            x: days,
            y: loansIssued,
            type: 'scatter',
            mode: 'lines',
            name: 'Loans Issued (Cumulative)'
        },
        {
            x: days,
            y: loansActive,
            type: 'scatter',
            mode: 'lines',
            name: 'Active Loans'
        },
        {
            x: days,
            y: loansDefaulted,
            type: 'scatter',
            mode: 'lines',
            name: 'Defaulted Loans (Cumulative)'
        },
        {
            x: days,
            y: loansRepaid,
            type: 'scatter',
            mode: 'lines',
            name: 'Repaid Loans (Cumulative)'
        }
    ], {
        title: 'Loan Performance',
        xaxis: { title: 'Simulation Day' },
        yaxis: { title: 'Number of Loans' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        legend: { orientation: 'h', y: 1.1 }
    });
}

// Data Explorer
function initializeDataExplorer() {
    // Sample data for demonstration
    var loanData = [
        { loan_id: 1, borrower_id: 101, trust_score: 0.85, loan_amount: 15000, interest_rate: 0.05, defaulted: 0 },
        { loan_id: 2, borrower_id: 102, trust_score: 0.65, loan_amount: 25000, interest_rate: 0.07, defaulted: 0 },
        { loan_id: 3, borrower_id: 103, trust_score: 0.45, loan_amount: 10000, interest_rate: 0.09, defaulted: 1 },
        { loan_id: 4, borrower_id: 104, trust_score: 0.92, loan_amount: 50000, interest_rate: 0.04, defaulted: 0 },
        { loan_id: 5, borrower_id: 105, trust_score: 0.35, loan_amount: 7500, interest_rate: 0.11, defaulted: 1 }
    ];
    
    var borrowerData = [
        { borrower_id: 101, income: 75000, debt_to_income: 0.25, age: 35, employment_length: 8 },
        { borrower_id: 102, income: 60000, debt_to_income: 0.30, age: 42, employment_length: 15 },
        { borrower_id: 103, income: 45000, debt_to_income: 0.40, age: 28, employment_length: 3 },
        { borrower_id: 104, income: 120000, debt_to_income: 0.20, age: 50, employment_length: 25 },
        { borrower_id: 105, income: 35000, debt_to_income: 0.45, age: 22, employment_length: 1 }
    ];
    
    var segmentStats = [
        { segment: 'High_Large', expected_return: 0.08, risk: 0.05, num_loans: 150, default_rate: 0.03 },
        { segment: 'Medium_Medium', expected_return: 0.10, risk: 0.07, num_loans: 300, default_rate: 0.05 },
        { segment: 'Low_Small', expected_return: 0.12, risk: 0.09, num_loans: 500, default_rate: 0.08 }
    ];
    
    // Populate dataset selector options
    var datasetSelector = document.getElementById('dataset-selector');
    datasetSelector.innerHTML = `
        <option value="loan_data">Loan Data</option>
        <option value="borrower_data">Borrower Data</option>
        <option value="segment_stats">Segment Statistics</option>
    `;
    
    // Populate visualization type selector
    var vizTypeSelector = document.getElementById('viz-type-selector');
    vizTypeSelector.innerHTML = `
        <option value="scatter">Scatter Plot</option>
        <option value="histogram">Histogram</option>
        <option value="box">Box Plot</option>
        <option value="bar">Bar Chart</option>
    `;
    
    // Populate axis selectors based on loan data (default)
    updateAxisOptions('loan_data');
    
    // Show data preview
    updateDataPreview('loan_data');
    
    // Create initial visualization
    updateDataExplorer();
}

function updateAxisOptions(dataset) {
    var columns = [];
    
    if (dataset === 'loan_data') {
        columns = ['loan_id', 'borrower_id', 'trust_score', 'loan_amount', 'interest_rate', 'defaulted'];
    } else if (dataset === 'borrower_data') {
        columns = ['borrower_id', 'income', 'debt_to_income', 'age', 'employment_length'];
    } else if (dataset === 'segment_stats') {
        columns = ['segment', 'expected_return', 'risk', 'num_loans', 'default_rate'];
    }
    
    // Update X-axis selector
    var xAxisSelector = document.getElementById('x-axis-selector');
    xAxisSelector.innerHTML = columns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    // Update Y-axis selector
    var yAxisSelector = document.getElementById('y-axis-selector');
    yAxisSelector.innerHTML = columns.map(col => `<option value="${col}">${col}</option>`).join('');
    if (yAxisSelector.options.length > 1) {
        yAxisSelector.selectedIndex = 1; // Select second option by default
    }
    
    // Update color selector
    var colorSelector = document.getElementById('color-selector');
    colorSelector.innerHTML = `<option value="none">None</option>` + 
                             columns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    // Update filter selector
    var filterSelector = document.getElementById('filter-selector');
    filterSelector.innerHTML = `<option value="none">None</option>` + 
                              columns.map(col => `<option value="${col}">${col}</option>`).join('');
}

function updateFilterControls() {
    var filterSelector = document.getElementById('filter-selector');
    var filterCol = filterSelector.value;
    
    if (filterCol === 'none') {
        document.getElementById('filter-controls').innerHTML = '';
        return;
    }
    
    // For demonstration, create a simple range filter
    document.getElementById('filter-controls').innerHTML = `
        <div>
            <label>Filter ${filterCol} Range:</label>
            <input type="range" id="filter-min" min="0" max="100" value="0" oninput="updateDataExplorer()">
            <span id="filter-min-value">0</span>
            <input type="range" id="filter-max" min="0" max="100" value="100" oninput="updateDataExplorer()">
            <span id="filter-max-value">100</span>
        </div>
    `;
    
    // Update filter value displays
    document.getElementById('filter-min').addEventListener('input', function() {
        document.getElementById('filter-min-value').textContent = this.value;
    });
    
    document.getElementById('filter-max').addEventListener('input', function() {
        document.getElementById('filter-max-value').textContent = this.value;
    });
}

function updateDataExplorer() {
    var dataset = document.getElementById('dataset-selector').value;
    var vizType = document.getElementById('viz-type-selector').value;
    var xCol = document.getElementById('x-axis-selector').value;
    var yCol = document.getElementById('y-axis-selector').value;
    var colorCol = document.getElementById('color-selector').value;
    
    // Sample data for demonstration
    var data = [];
    if (dataset === 'loan_data') {
        data = [
            { loan_id: 1, borrower_id: 101, trust_score: 0.85, loan_amount: 15000, interest_rate: 0.05, defaulted: 0 },
            { loan_id: 2, borrower_id: 102, trust_score: 0.65, loan_amount: 25000, interest_rate: 0.07, defaulted: 0 },
            { loan_id: 3, borrower_id: 103, trust_score: 0.45, loan_amount: 10000, interest_rate: 0.09, defaulted: 1 },
            { loan_id: 4, borrower_id: 104, trust_score: 0.92, loan_amount: 50000, interest_rate: 0.04, defaulted: 0 },
            { loan_id: 5, borrower_id: 105, trust_score: 0.35, loan_amount: 7500, interest_rate: 0.11, defaulted: 1 }
        ];
    } else if (dataset === 'borrower_data') {
        data = [
            { borrower_id: 101, income: 75000, debt_to_income: 0.25, age: 35, employment_length: 8 },
            { borrower_id: 102, income: 60000, debt_to_income: 0.30, age: 42, employment_length: 15 },
            { borrower_id: 103, income: 45000, debt_to_income: 0.40, age: 28, employment_length: 3 },
            { borrower_id: 104, income: 120000, debt_to_income: 0.20, age: 50, employment_length: 25 },
            { borrower_id: 105, income: 35000, debt_to_income: 0.45, age: 22, employment_length: 1 }
        ];
    } else if (dataset === 'segment_stats') {
        data = [
            { segment: 'High_Large', expected_return: 0.08, risk: 0.05, num_loans: 150, default_rate: 0.03 },
            { segment: 'Medium_Medium', expected_return: 0.10, risk: 0.07, num_loans: 300, default_rate: 0.05 },
            { segment: 'Low_Small', expected_return: 0.12, risk: 0.09, num_loans: 500, default_rate: 0.08 }
        ];
    }
    
    // Extract x and y values
    var x = data.map(d => d[xCol]);
    var y = vizType !== 'histogram' ? data.map(d => d[yCol]) : null;
    
    // Extract color values if specified
    var color = colorCol !== 'none' ? data.map(d => d[colorCol]) : null;
    
    // Create appropriate plot based on visualization type
    var plotData = [];
    
    if (vizType === 'scatter') {
        plotData = [{
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: color,
                colorscale: 'Viridis',
                showscale: color !== null
            },
            text: data.map((d, i) => {
                var text = '';
                for (var key in d) {
                    text += `${key}: ${d[key]}<br>`;
                }
                return text;
            }),
            hoverinfo: 'text'
        }];
    } else if (vizType === 'histogram') {
        plotData = [{
            x: x,
            type: 'histogram',
            marker: {
                color: 'rgba(52, 152, 219, 0.7)',
                line: {
                    color: 'rgba(52, 152, 219, 1)',
                    width: 1
                }
            }
        }];
    } else if (vizType === 'box') {
        plotData = [{
            y: y,
            x: x,
            type: 'box',
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: -1.8
        }];
    } else if (vizType === 'bar') {
        plotData = [{
            x: x,
            y: y,
            type: 'bar',
            marker: {
                color: color,
                colorscale: 'Viridis',
                showscale: color !== null
            }
        }];
    }
    
    // Create layout
    var layout = {
        title: `${dataset} - ${vizType}`,
        xaxis: { title: xCol },
        yaxis: { title: yCol },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    };
    
    // Create plot
    Plotly.newPlot('data-explorer-graph', plotData, layout);
}

function updateDataPreview(dataset) {
    var previewDiv = document.getElementById('data-preview');
    var html = '';
    
    if (dataset === 'loan_data') {
        html = `
            <h4>Loan Data Preview (First 5 rows)</h4>
            <div class="table-container">
                <table>
                    <tr>
                        <th>loan_id</th>
                        <th>borrower_id</th>
                        <th>trust_score</th>
                        <th>loan_amount</th>
                        <th>interest_rate</th>
                        <th>defaulted</th>
                    </tr>
                    <tr><td>1</td><td>101</td><td>0.85</td><td>15000</td><td>0.05</td><td>0</td></tr>
                    <tr><td>2</td><td>102</td><td>0.65</td><td>25000</td><td>0.07</td><td>0</td></tr>
                    <tr><td>3</td><td>103</td><td>0.45</td><td>10000</td><td>0.09</td><td>1</td></tr>
                    <tr><td>4</td><td>104</td><td>0.92</td><td>50000</td><td>0.04</td><td>0</td></tr>
                    <tr><td>5</td><td>105</td><td>0.35</td><td>7500</td><td>0.11</td><td>1</td></tr>
                </table>
            </div>
        `;
    } else if (dataset === 'borrower_data') {
        html = `
            <h4>Borrower Data Preview (First 5 rows)</h4>
            <div class="table-container">
                <table>
                    <tr>
                        <th>borrower_id</th>
                        <th>income</th>
                        <th>debt_to_income</th>
                        <th>age</th>
                        <th>employment_length</th>
                    </tr>
                    <tr><td>101</td><td>75000</td><td>0.25</td><td>35</td><td>8</td></tr>
                    <tr><td>102</td><td>60000</td><td>0.30</td><td>42</td><td>15</td></tr>
                    <tr><td>103</td><td>45000</td><td>0.40</td><td>28</td><td>3</td></tr>
                    <tr><td>104</td><td>120000</td><td>0.20</td><td>50</td><td>25</td></tr>
                    <tr><td>105</td><td>35000</td><td>0.45</td><td>22</td><td>1</td></tr>
                </table>
            </div>
        `;
    } else if (dataset === 'segment_stats') {
        html = `
            <h4>Segment Statistics Preview (First 3 rows)</h4>
            <div class="table-container">
                <table>
                    <tr>
                        <th>segment</th>
                        <th>expected_return</th>
                        <th>risk</th>
                        <th>num_loans</th>
                        <th>default_rate</th>
                    </tr>
                    <tr><td>High_Large</td><td>0.08</td><td>0.05</td><td>150</td><td>0.03</td></tr>
                    <tr><td>Medium_Medium</td><td>0.10</td><td>0.07</td><td>300</td><td>0.05</td></tr>
                    <tr><td>Low_Small</td><td>0.12</td><td>0.09</td><td>500</td><td>0.08</td></tr>
                </table>
            </div>
        `;
    }
    
    previewDiv.innerHTML = html;
}

// Add CSS to style the tabs
document.addEventListener('DOMContentLoaded', function() {
    var style = document.createElement('style');
    style.textContent = `
        .tabs-navigation {
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 20px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .tab-button {
            background-color: #f8f9fa;
            border: none;
            padding: 10px 20px;
            margin-right: 5px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .tab-button:hover {
            background-color: #e9ecef;
        }
        
        .tab-button.active {
            background-color: #3498db;
            color: white;
        }
        
        .tab-content {
            display: none;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .tab-content.active {
            display: block;
        }
        
        .chart-container {
            height: 400px;
            margin-bottom: 20px;
        }
    `;
    document.head.appendChild(style);
});
