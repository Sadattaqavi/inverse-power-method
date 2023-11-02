import numpy as np

def inverse_power_method(A, initial_vector, epsilon=1e-6, max_iterations=100):
    n = A.shape[0]
    x = initial_vector / np.linalg.norm(initial_vector)  # Normalize the initial vector

    for i in range(max_iterations):
        y = np.linalg.solve(A, x)
        eigenvalue = np.max(np.abs(y))
        x = y / eigenvalue

        # Print the iteration details for the first 5 iterations
        if i < 5:
            print(f"Iteration {i + 1}:")
            print("Vector:", np.round(y, decimals=4))
            print("Normalized Vector:", np.round(x, decimals=4))
            print()

        if np.linalg.norm(A @ x - eigenvalue * x) < epsilon:
            break

    eigenvalue = 1 / eigenvalue
    return eigenvalue, x
A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 4]])

initial_vector = np.array([1, 0, 0])

eigenvalue, eigenvector = inverse_power_method(A, initial_vector)

print("Inverse Eigenvalue:", round(eigenvalue, 4))
print("Eigenvector:", np.round(eigenvector, decimals=4))