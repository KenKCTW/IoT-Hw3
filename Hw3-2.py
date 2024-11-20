import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to generate and plot the points
def generate_and_plot_points():
    # Step 1: Generate 600 random 2D points with a normal distribution
    mean = [0, 0]
    variance = 10
    num_points = 600

    # Generate random points from a normal distribution
    points = np.random.multivariate_normal(mean, np.diag([variance, variance]), num_points)

    # Separate the points into x and y coordinates
    x_points = points[:, 0]
    y_points = points[:, 1]

    # Step 2: Calculate the distance from the origin (0,0)
    distances = np.sqrt(x_points**2 + y_points**2)

    # Step 3: Assign labels based on the distance
    labels = np.where(distances < 4, 0, 1)  # Y=0 if distance < 4, Y=1 if distance >= 4

    # Step 4: Create scatter plot
    plt.figure(figsize=(6, 6))

    # Plot points with Y=0 (distance < 4) in blue
    plt.scatter(x_points[labels == 0], y_points[labels == 0], color='blue', label='Y=0 (distance < 4)', alpha=0.6)

    # Plot points with Y=1 (distance >= 4) in red
    plt.scatter(x_points[labels == 1], y_points[labels == 1], color='red', label='Y=1 (distance >= 4)', alpha=0.6)

    # Add labels and title
    plt.title("Scatter Plot of Random Points with Labels Based on Distance from Origin")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # Show the plot in Streamlit
    st.pyplot(plt)

# Streamlit app interface
st.title("Random Points Generation and Labeling")

st.write(
    """
    This app generates 600 random 2D points, calculates the distance of each point from the origin (0,0),
    assigns labels based on the distance (Y=0 for distance < 4, Y=1 for distance >= 4),
    and displays a scatter plot showing the results.
    """
)

# Button to generate and display points
if st.button('Generate Random Points and Plot'):
    generate_and_plot_points()
