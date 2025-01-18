import numpy as np        
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
from math import degrees, cos, pi
import cmath
import random

#----------------------------------------------------------------

# Question 1 & 2 & 3

def get_area(p1, p2, p3):
    # Convert the points to numpy arrays for easier calculations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate the side lengths of the triangle
    side_12 = np.linalg.norm(p2 - p1)
    side_13 = np.linalg.norm(p3 - p1)
    side_23 = np.linalg.norm(p3 - p2)

    # Calculate the semi-perimeter of the triangle
    s = (side_12 + side_13 + side_23) / 2

    # Calculate the area of the triangle using Heron's formula
    area = np.sqrt(s * (s - side_12) * (s - side_13) * (s - side_23))

    return area

def plot_triangle(p1, p2, p3):
    
    # Convert the points to numpy arrays for easier calculations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate the side lengths of the triangle
    side_12 = np.linalg.norm(p2 - p1)
    side_13 = np.linalg.norm(p3 - p1)
    side_23 = np.linalg.norm(p3 - p2)

    # Define the start and end points of each side of the triangle
    side_12_start = p1
    side_12_end = p2
    side_13_start = p1
    side_13_end = p3
    side_23_start = p2
    side_23_end = p3

    # Calculate all sides
    ab = p2 - p1
    ac = p3 - p1
    ba = p1 - p2
    bc = p3 - p2
    ca = p1 - p3
    cb = p2 - p3
    
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the vertices of the triangle
    vertices = np.array([p1, p2, p3])

    # Define the triangles that make up the surface of the triangle
    triangles = np.array([[0, 1, 2]])

    # Plot the side lengths of the triangle as lines
    ax.plot([side_12_start[0], side_12_end[0]], [side_12_start[1], side_12_end[1]], [side_12_start[2], side_12_end[2]], color='green')
    ax.plot([side_13_start[0], side_13_end[0]], [side_13_start[1], side_13_end[1]], [side_13_start[2], side_13_end[2]], color='green')
    ax.plot([side_23_start[0], side_23_end[0]], [side_23_start[1], side_23_end[1]], [side_23_start[2], side_23_end[2]], color='green')

    # Plot the triangle as a surface and side edges
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, color='red', alpha=0.5)

    # Calculate the interior angles of the triangle
    angle_p1 = np.degrees(np.arccos(np.dot(ab , ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))))
    angle_p2 = np.degrees(np.arccos(np.dot(ba , bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))
    angle_p3 = np.degrees(np.arccos(np.dot(ca , cb) / (np.linalg.norm(ca) * np.linalg.norm(cb))))

    # Add labels at each vertex with the vertex name and interior angle
    ax.text(p1[0], p1[1], p1[2], f"p1 ({angle_p1:.2f}°)", color='blue')
    ax.text(p2[0], p2[1], p2[2], f"p2 ({angle_p2:.2f}°)", color='blue')
    ax.text(p3[0], p3[1], p3[2], f"p3 ({angle_p3:.2f}°)", color='blue')


    # Set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

def main():

    p1 = [0, 0, 0]
    p2 = [1, 1, 1]
    p3 = [1, 1, 0]

    area = get_area(p1, p2, p3)
    print(f'The Area of the triangle is: {area}')

    plot_triangle(p1, p2, p3)   

if __name__ == "__main__":
    main()

#----------------------------------------------------------------

# Question 4

def polygon_approximation_circle(N):
    # Generate N points uniformly spaced around the circumference
    theta = np.linspace(0, 2 * pi, N, endpoint=False)  # Set endpoint=False for N unique vertices
    x = np.cos(theta)
    y = np.sin(theta)

    # Calculate the area of the polygon approximation
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))  # Shoelace formula

    return area

def main():
    # Define the values of N
    N_values = [4, 10, 20, 40, 100]

    # Calculate the area of the polygon approximation for each N value
    areas = [polygon_approximation_circle(N) for N in N_values]

    # Print the areas
    print("Areas of the polygon approximations:")
    for N, area in zip(N_values, areas):
        print(f"N = {N}: Area = {area:.4f}")

    # Plot the polygon approximations
    fig, ax = plt.subplots()
    for N in N_values:
        theta = np.linspace(0, 2 * pi, N, endpoint=False)  # Set endpoint=False for N unique vertices
        x = np.cos(theta)
        y = np.sin(theta)

        # Explicitly close the polygon by appending the first point to the end
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        ax.plot(x, y, label=f"N = {N}")
    ax.legend()
    ax.set_aspect('equal')
    ax.set_title("Polygon Approximations to a Circle")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()

#----------------------------------------------------------------

# Question 5

# Define the matrices A and B
A = np.array([[1, 2, -3],
              [4, 8, 8],
              [2, 2, 4]])

B = np.array([[5, 5, -3],
              [4, 8, 8],
              [2, 2, 4]])

# Create the block matrix using np.block
block_matrix = np.block([[A, B], [np.zeros((3, 3)), A]])

# Define the right-hand side vector
rhs_vector = np.array([1, 0, 0, 0, 0, 0])

# Solve the system of linear equations
solution = np.linalg.solve(block_matrix, rhs_vector)

# Print the solution
print("The solution vector x is:")
print(solution)
