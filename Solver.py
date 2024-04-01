
import numpy as np
from scipy.sparse import csr_matrix
def update_y(x_bound, bounds, y, L, learning_rate, alpha):
    # Ensure L is a CSR matrix for efficient matrix-vector multiplication
    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)
    
    # Compute the base gradient for all elements as influenced by L
    grad = 2 * alpha * L.dot(y)
    
    # Update the gradient for the boundary elements
    for i, bound_idx in enumerate(bounds):
        # grad[bound_idx] = -2 * (x_bound[i] - y[bound_idx])
        grad[bound_idx] =0
    
    # Update y using the gradient
    y -= learning_rate * grad
    return y

def compute_loss(x_bound, bounds, y, L, alpha):
    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)
    
    # Compute the loss function value
    term1 = np.sum((x_bound - y[bounds]) ** 2)
    term2 = alpha * y.T.dot(L.dot(y))
    loss = term1 + term2
    return loss

def Solve(x_bounds, bounds, L, alpha, learning_rate=0.01, num_iter=100, convergence_threshold=1e-3, verbose=False,loss=False):
    y = np.zeros(L.shape[0])
    y[bounds] = x_bounds
    prev_loss = np.inf

    if loss:
        losses = []

    for iteration in range(num_iter):
        y = update_y(x_bounds, bounds, y, L, learning_rate, alpha)
        loss = compute_loss(x_bounds, bounds, y, L, alpha)
        if loss > prev_loss:
            alpha /=10
        if verbose and iteration % 10 == 0:
            print(f'Iteration: {iteration}, Loss: {loss}')

        if np.abs(loss - prev_loss) < convergence_threshold: 
            print(f"Converged after {iteration} iterations with loss: {loss}")
            break
        if iteration == num_iter - 1:
            print(f"Did not converge after {num_iter} iterations with loss: {loss}")
        prev_loss = loss
        if loss:
            losses.append(loss)
    
    return y,losses


