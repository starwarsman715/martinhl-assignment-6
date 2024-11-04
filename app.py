from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # STEP 1
    # Generate random dataset X and Y
    X = np.random.uniform(0, 1, N)  # Random values between 0 and 1
    X = X.reshape(-1, 1)  # Reshape for sklearn
    
    # Generate Y with normal additive error
    Y = mu + np.random.normal(0, np.sqrt(sigma2), N)  # Normal distribution with mean mu and variance sigma2
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='blue', alpha=0.5, label='Data points')
    plt.plot(X, model.predict(X), color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression with Random Data')
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()
    
    # Step 2: Run S simulations
    slopes = []
    intercepts = []
    
    for _ in range(S):
        # Generate new random datasets for each simulation
        X_sim = np.random.uniform(0, 1, N).reshape(-1, 1)
        Y_sim = mu + np.random.normal(0, np.sqrt(sigma2), N)
        
        # Fit linear regression
        sim_model = LinearRegression()
        sim_model.fit(X_sim, Y_sim)
        
        # Store results
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme  # Fixed this line

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)