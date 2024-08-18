# **Monte Carlo Simulation for Stock Prices**

This application uses Monte Carlo methods to simulate and analyze future stock prices based on historical data. By employing Geometric Brownian Motion (GBM), the tool generates multiple potential price paths and provides a visual representation through animated plots. Users can input stock parameters, view simulation results, and export data for further analysis. Ideal for educational purposes, this tool helps in understanding stock price dynamics and risk assessments.


## Features
- **Stock Price Simulation**: Simulates future stock price paths using GBM.
- **Visualization**: Animated stock price paths and summary statistics plots.
- **Summary Statistics**: Mean price, confidence intervals, standard deviation.
- **CSV Export**: Save simulation results and statistics to CSV files.
- **GUI Interface**: User-friendly GUI built with Tkinter.


## Installation
To get started, ensure you have Python 3.x installed along with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `yfinance`
- `tkinter` (usually included with Python)

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn yfinance
```

## Usage

### Running the Application

1. **Launch the Application**  
   Execute the script to open the main GUI.

2. **Input Parameters**  
   - Click **"Start Simulation"** to access the input parameters window.  
   - Enter the following details:
      - **Stock Ticker Symbol**: e.g., AAPL
      - **Start Date**: Format `YYYY-MM-DD`
      - **End Date**: Format `YYYY-MM-DD`
      - **Number of Time Steps**: e.g., 300
      - **Number of Paths**: e.g., 100
   
   - Click **"Run Simulation"** to initiate the process.

3. **Viewing Results**  
   - Observe animated stock price paths.
   - Review summary statistics, including mean price, confidence intervals, and standard deviation.
   - View the final simulation plot.
   
      **NOTE** : There are no next or previous buttons to toggle between visualizations; you must close the current window to view the next visualization.

4. **Exporting Data**  
   Click **"Save to CSV"** to export the simulation data and summary statistics.

5. **About**  
   Click **"About"** to learn more about the Monte Carlo simulation and its usage.

### GUI Layout
1. **Main Window**
    - **About**: Opens the About window.
    - **Start Simulation**: Opens the input parameters window.
    - **Exit**: Closes the application.

2. **Input Parameters Window**

    Fields for entering stock ticker symbol, start and end dates, number of time steps, and number of paths.
    - **Run Simulation**: Starts the simulation process.
    - **Cancel**: Closes the window.

3. **Simulation Summary Window**
   
    Displays summary statistics and investment recommendations.
    - **Save to CSV**: Saves results to CSV files.
    - **Run Another Simulation**: Opens the input parameters window for a new simulation.
    - **Exit**: Closes the summary window.

4. **About Window**

    Provides an overview of Monte Carlo simulation, its purpose, user instructions, and disclaimer.

## Disclaimer

This simulation is for educational and informational purposes only. It does not constitute financial advice. Consult a financial advisor before making any investment decisions.
