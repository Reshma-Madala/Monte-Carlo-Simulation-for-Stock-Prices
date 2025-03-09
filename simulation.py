import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import tkinter as tk
from tkinter import simpledialog, messagebox, Label, Entry, Button
from tkinter.filedialog import asksaveasfilename
from datetime import datetime
from yahooquery import Ticker
import requests

# Function to calculate log returns
def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1))

# Function to simulate future stock prices using GBM
def simulate_stock_prices(S0, mu, sigma, T, dt, N):
    Z = np.random.standard_normal(size=(T, N))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    price_paths = np.zeros_like(Z)
    price_paths[0] = S0
    for t in range(1, T):
        price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion[t])
    return price_paths

# Function to animate the simulated stock price paths
def animate_simulation(price_paths, S0, T, interval=20, title='Time-Lapse of Simulated Stock Price Paths', company_name=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if company_name:
        ax.set_title(f"{company_name} - {title}", fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.set_xlim(0, T)
    ax.set_ylim(price_paths.min(), price_paths.max())
    ax.axhline(y=S0, color='r', linestyle='--', label=f'Starting Price: {S0}')
    ax.legend()
    
    lines = [ax.plot([], [], lw=1)[0] for _ in range(price_paths.shape[1])]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        for j, line in enumerate(lines):
            line.set_data(np.arange(i), price_paths[:i, j])
        return lines

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=T, interval=interval, blit=True)
    
    # Run the animation three times
    for _ in range(3):
        plt.show()
    plt.close(fig)

# Function to calculate and plot summary statistics
def plot_summary_statistics(price_paths, T, N, confidence_interval=0.95, company_name=None):
    final_prices = price_paths[-1]
    mean_price = np.mean(final_prices)
    lower_bound = np.percentile(final_prices, (1 - confidence_interval) / 2 * 100)
    upper_bound = np.percentile(final_prices, (1 + confidence_interval) / 2 * 100)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(final_prices, kde=True, bins=50)
    plt.axvline(x=mean_price, color='r', linestyle='--', label=f'Mean: {mean_price:.2f}')
    plt.axvline(x=lower_bound, color='g', linestyle='--', label=f'Lower {int(confidence_interval*100)}% CI: {lower_bound:.2f}')
    plt.axvline(x=upper_bound, color='g', linestyle='--', label=f'Upper {int(confidence_interval*100)}% CI: {upper_bound:.2f}')
    
    if company_name:
        plt.title(f'{company_name} - Histogram of Simulated Final Stock Prices (T={T}, N={N})', fontsize=14)
    else:
        plt.title(f'Histogram of Simulated Final Stock Prices (T={T}, N={N})', fontsize=14)
    
    plt.xlabel('Stock Price', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.show()
    
    return mean_price, lower_bound, upper_bound, np.min(final_prices), np.max(final_prices), np.std(final_prices)

# Function to plot the final simulation picture
def plot_final_simulation(price_paths, company_name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(price_paths)
    
    if company_name:
        plt.title(f'{company_name} - Final Simulated Stock Price Paths', fontsize=14)
    else:
        plt.title('Final Simulated Stock Price Paths', fontsize=14)
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.show()

# Function to save simulation data to CSV
def save_to_csv(price_paths, summary_stats, prices, company_name=None):
    
    save_dir = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Simulation Files")
    
    if not save_dir:
        return  

    try:
        price_paths_df = pd.DataFrame(price_paths)
        price_paths_file = save_dir.replace('.csv', '_price_paths.csv')
        price_paths_df.to_csv(price_paths_file, index=False)
        
        summary_df = pd.DataFrame([summary_stats], columns=['Mean Price', 'Lower CI', 'Upper CI', 'Worst Case', 'Best Case', 'Standard Deviation'])
        summary_file = save_dir.replace('.csv', '_summary_stats.csv')
        summary_df.to_csv(summary_file, index=False)
       
        prices_df = pd.DataFrame(prices)
        prices_file = save_dir.replace('.csv', '_original_prices.csv')
        prices_df.to_csv(prices_file)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the files: {str(e)}")

# Function to display the simulation summary in a separate window
def display_simulation_summary(mean_price, lower_bound, upper_bound, worst_case, best_case, std_dev, S0, price_paths, prices, company_name=None):
    try:
        summary_window = tk.Toplevel()
        
        if company_name:
            summary_window.title(f"{company_name} - Simulation Summary")
            title_text = f"{company_name} - Monte Carlo Simulation Summary"
        else:
            summary_window.title("Simulation Summary")
            title_text = "Monte Carlo Simulation Summary"
        
        summary_window.geometry("700x700")

        Label(summary_window, text=title_text, font=("Georgia", 16)).pack(pady=10)

        Label(summary_window, text=f"Expected Mean Price: ${mean_price:.2f}", font=("Georgia", 12)).pack(pady=5)
        Label(summary_window, text=f"95% Confidence Interval: (${lower_bound:.2f} to ${upper_bound:.2f})", font=("Georgia", 12)).pack(pady=5)
        Label(summary_window, text=f"Worst-case Scenario: ${worst_case:.2f}", font=("Georgia", 12)).pack(pady=5)
        Label(summary_window, text=f"Best-case Scenario: ${best_case:.2f}", font=("Georgia", 12)).pack(pady=5)
        Label(summary_window, text=f"Standard Deviation of Final Prices: ${std_dev:.2f}", font=("Georgia", 12)).pack(pady=5)

        roi_dollars = mean_price - S0  # ROI in dollars
        roi_percentage = (roi_dollars / S0) * 100  # ROI as a percentage
        Label(summary_window, text=f"ROI in Dollars: ${roi_dollars:.2f}", font=("Georgia", 12)).pack(pady=5)
        Label(summary_window, text=f"ROI Percentage: {roi_percentage:.2f}%", font=("Georgia", 12)).pack(pady=5)

        # Updated investment decision logic that's more aligned with the analysis summary
        if roi_percentage > 15:
            investment_decision = ("Based on this simulation, the stock shows strong potential for significant returns. "
                                  "The mean expected price is well above the starting price, suggesting a favorable outlook.")
        elif roi_percentage > 0:
            investment_decision = ("The simulation indicates a moderately positive outlook with the mean price above the starting price. "
                                  "The investment might be worthwhile if the risk level aligns with your tolerance.")
        elif roi_percentage > -10:
            investment_decision = ("The simulation suggests a slightly negative outlook. Consider your risk tolerance "
                                  "and perhaps explore additional analysis before investing.")
        else:
            investment_decision = ("The simulation indicates a strong negative outlook with significant potential downside. "
                                  "It may be advisable to seek alternative investment opportunities.")

        # Updated decision rationale to align better with the investment decision
        decision_rationale = (f"Analysis is based on {price_paths.shape[1]} simulated paths over {price_paths.shape[0]} time steps. "
                             f"The mean price projection is ${mean_price:.2f} (${roi_dollars:.2f} or {roi_percentage:.2f}% from starting price). "
                             f"The 95% confidence interval ranges from ${lower_bound:.2f} to ${upper_bound:.2f}, "
                             f"with worst and best case scenarios of ${worst_case:.2f} and ${best_case:.2f} respectively. "
                             f"The standard deviation of ${std_dev:.2f} indicates the level of volatility in the projections.")

        Label(summary_window, text="Investment Recommendation", font=("Georgia", 14)).pack(pady=10)
        Label(summary_window, text=investment_decision, font=("Georgia", 12), wraplength=480).pack(pady=5)

        Label(summary_window, text="Analysis Summary", font=("Georgia", 14)).pack(pady=10)
        Label(summary_window, text=decision_rationale, font=("Georgia", 12), wraplength=480).pack(pady=5)

        button_frame = tk.Frame(summary_window)
        button_frame.pack(pady=20)

        save_button = Button(button_frame, text="Save to CSV", command=lambda: save_to_csv(price_paths, (mean_price, lower_bound, upper_bound, worst_case, best_case, std_dev), prices, company_name), font=("Georgia", 12))
        save_button.pack(side=tk.LEFT, padx=10)

        ok_button = Button(button_frame, text="OK", command=lambda: [summary_window.destroy(), show_options_window()], font=("Georgia", 12))
        ok_button.pack(side=tk.LEFT, padx=10)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while displaying the summary: {str(e)}")

def get_company_name(ticker_symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker_symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'quotes' in data and len(data['quotes']) > 0:
                for quote in data['quotes']:
                    if quote['symbol'].upper() == ticker_symbol.upper():
                        return quote['longname'] or quote['shortname'] or ticker_symbol
            
        # Fallback to the autocomplete API if the first method fails
        url_autocomplete = f"https://query2.finance.yahoo.com/v6/finance/autocomplete?query={ticker_symbol}&lang=en"
        response_auto = requests.get(url_autocomplete, headers=headers)
        
        if response_auto.status_code == 200:
            data_auto = response_auto.json()
            
            if 'ResultSet' in data_auto and 'Result' in data_auto['ResultSet']:
                for result in data_auto['ResultSet']['Result']:
                    if result['symbol'].upper() == ticker_symbol.upper():
                        return result.get('name', ticker_symbol)
        
        return ticker_symbol
        
    except Exception as e:
        print(f"Error retrieving company name: {e}")
        return ticker_symbol

# Main function to perform Monte Carlo simulation with separate animation and statistics
def monte_carlo_simulation(prices, T=300, N=100, confidence_interval=0.95, interval=20, company_name=None):
    try:
        log_returns = calculate_log_returns(prices)
        mu = log_returns.mean() * T  
        sigma = log_returns.std() * np.sqrt(T) 
        S0 = prices.iloc[-1]  # Starting price
        dt = 1 / T  # Time step
        price_paths = simulate_stock_prices(S0, mu, sigma, T, dt, N)

        animate_simulation(price_paths, S0, T, interval=interval, company_name=company_name)

        mean_price, lower_bound, upper_bound, worst_case, best_case, std_dev = plot_summary_statistics(price_paths, T, N, confidence_interval, company_name)
      
        plot_final_simulation(price_paths, company_name)

        display_simulation_summary(mean_price, lower_bound, upper_bound, worst_case, best_case, std_dev, S0, price_paths, prices, company_name)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during the simulation: {str(e)}")

# Function to handle "About" button click
def about():
    about_window = tk.Toplevel()
    about_window.title("About Monte Carlo Simulation")
    about_window.geometry("650x800")

    Label(about_window, text="Monte Carlo Simulation", font=("Georgia", 18, 'bold')).pack(pady=15)

    Label(about_window, text=(
        "Monte Carlo Simulation is a powerful mathematical technique used to model the probability of different outcomes "
        "in a process that cannot easily be predicted due to the intervention of random variables. Named after the Monte Carlo Casino "
        "due to the element of chance involved, it is widely used in various fields including finance, engineering, and project management."
    ), font=("Georgia", 12), wraplength=600, justify=tk.LEFT).pack(pady=10, padx=15)

    Label(about_window, text=(
        "This tool provides a user-friendly interface to perform Monte Carlo simulations for stock price forecasting. Key features include:\n"
        "- Simulating multiple future price paths for a stock using Geometric Brownian Motion (GBM).\n"
        "- Calculating and displaying summary statistics such as mean price, confidence intervals, and standard deviation.\n"
        "- Providing visualizations including animated price paths and histograms of final prices.\n"
        "- Allowing users to save simulation results and summary statistics to CSV files."
    ), font=("Georgia", 12), wraplength=600, justify=tk.LEFT).pack(pady=10, padx=15)

    Label(about_window, text=(
        "User Instructions:\n"
        "1. Enter the stock ticker symbol, start and end dates, number of time steps, and number of paths.\n"
        "2. Run the simulation to generate and visualize multiple price paths.\n"
        "3. Review the summary statistics and decide on investment strategies based on the results."
    ), font=("Georgia", 12), wraplength=600, justify=tk.LEFT).pack(pady=10, padx=15)

    Label(about_window, text=(
        "Disclaimer:\n"
        "This simulation is intended for educational and informational purposes only. It does not constitute financial advice. "
        "Please consult with a financial advisor before making any investment decisions."
    ), font=("Georgia", 12), wraplength=600, justify=tk.LEFT).pack(pady=10, padx=15)

    button_frame = tk.Frame(about_window)
    button_frame.pack(pady=20)
    
    Button(button_frame, text="Start Simulation", command=input_parameters, font=("Georgia", 14)).pack(side=tk.LEFT, padx=10)
    Button(button_frame, text="Back", command=about_window.destroy, font=("Georgia", 14)).pack(side=tk.LEFT, padx=10)

# Function to handle the "Start Simulation" button click
def input_parameters():
    input_window = tk.Toplevel()
    input_window.title("Monte Carlo Simulation Input")
    input_window.geometry("600x600")
    
    Label(input_window, text="Input Parameters for Monte Carlo Simulation", font=("Georgia", 16)).pack(pady=10)
    
    Label(input_window, text="Stock Ticker Symbol:", font=("Georgia", 12)).pack(pady=5)
    ticker_entry = Entry(input_window, font=("Georgia", 12))
    ticker_entry.pack(pady=5)
    
    Label(input_window, text="Start Date (YYYY-MM-DD):", font=("Georgia", 12)).pack(pady=5)
    start_date_entry = Entry(input_window, font=("Georgia", 12))
    start_date_entry.pack(pady=5)
    
    Label(input_window, text="End Date (YYYY-MM-DD):", font=("Georgia", 12)).pack(pady=5)
    end_date_entry = Entry(input_window, font=("Georgia", 12))
    end_date_entry.pack(pady=5)
    
    Label(input_window, text="Number of Time Steps (e.g., 300):", font=("Georgia", 12)).pack(pady=5)
    time_steps_entry = Entry(input_window, font=("Georgia", 12))
    time_steps_entry.pack(pady=5)
    
    Label(input_window, text="Number of Paths (e.g., 100):", font=("Georgia", 12)).pack(pady=5)
    num_paths_entry = Entry(input_window, font=("Georgia", 12))
    num_paths_entry.pack(pady=5)
    
    def run_simulation():
        ticker_symbol = ticker_entry.get()
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.")
            return

        try:
            time_steps = int(time_steps_entry.get())
            num_paths = int(num_paths_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Number of time steps and paths must be integers.")
            return

        try:
            # Get company name
            company_name = get_company_name(ticker_symbol)
            
            # Using yahooquery instead of yfinance
            ticker = Ticker(ticker_symbol)
            history = ticker.history(start=start_date, end=end_date)
            
            if history.empty:
                messagebox.showerror("Error", "Invalid ticker symbol or no data available for the given date range.")
                return
                
            # Reshape the data to match the expected format
            # yahooquery returns a multi-index DataFrame
            # We need to extract the 'adjclose' column for the specified ticker
            if isinstance(history.index, pd.MultiIndex):
                # Handle case where multiple symbols might be returned
                prices = history.xs(ticker_symbol, level='symbol')['adjclose'] if ticker_symbol in history.index.get_level_values('symbol') else None
                
                if prices is None or prices.empty:
                    # If the specific ticker isn't found, try to get the first available symbol
                    available_symbols = history.index.get_level_values('symbol').unique()
                    if len(available_symbols) > 0:
                        prices = history.xs(available_symbols[0], level='symbol')['adjclose']
                    else:
                        raise ValueError("No data available for the ticker")
            else:
                # Handle case where only a single ticker is returned (no multi-index)
                prices = history['adjclose']
            
            # Make sure we have enough data for the simulation
            if len(prices) < 30:  # Arbitrary minimum number of data points
                messagebox.showerror("Error", "Not enough historical data for a reliable simulation. Please extend the date range.")
                return
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while fetching stock data: {str(e)}")
            return

        monte_carlo_simulation(prices, T=time_steps, N=num_paths, confidence_interval=0.95, interval=20, company_name=company_name)
        
        input_window.destroy()

    Button(input_window, text="Run Simulation", command=run_simulation, font=("Georgia", 14)).pack(pady=20)
    Button(input_window, text="Cancel", command=input_window.destroy, font=("Georgia", 14)).pack(pady=10)

# Function to handle the "Exit" button click
def exit_app():
    main_window.destroy()

# Function to show options after simulation ends
def show_options_window():
    options_window = tk.Toplevel()
    options_window.title("Simulation Complete")
    options_window.geometry("300x200")
    
    Label(options_window, text="Simulation Complete", font=("Georgia", 16)).pack(pady=10)
    Button(options_window, text="Run Another Simulation", command=input_parameters, font=("Georgia", 14)).pack(pady=10)
    Button(options_window, text="Exit", command=exit_app, font=("Georgia", 14)).pack(pady=10)

# Main GUI window
try:
    main_window = tk.Tk()
    main_window.title("Monte Carlo Simulation")
    main_window.geometry("400x300")

    Label(main_window, text="Monte Carlo Simulation", font=("Georgia", 18)).pack(pady=20)

    Button(main_window, text="About", command=about, font=("Georgia", 14)).pack(pady=10)
    Button(main_window, text="Start Simulation", command=input_parameters, font=("Georgia", 14)).pack(pady=10)
    Button(main_window, text="Exit", command=exit_app, font=("Georgia", 14)).pack(pady=10)

    main_window.mainloop()
except Exception as e:
    messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")