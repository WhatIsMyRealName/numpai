import numpy as np
import scipy.special

# No documentation here. (https://i.redd.it/trf5lvmxe6x11.jpg)

# tanh activation function and its derivative
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1-np.tanh(x)**2

# ReLU activation function, its derivative, and similars
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_prime(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def leakyrelu(x: np.ndarray, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leakyrelu_prime(x: np.ndarray, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, 1, alpha)

# GELU and its derivatives
def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Computes the exact GELU function. You probably don't need as much accuracy and it might slow down your programm."""
    return x * 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))

def gelu_exact_prime(x: np.ndarray) -> np.ndarray:
    """Computes the exact derivatives of GELU. You probably don't need as much accuracy and it might slow down tour programm."""
    phi_x = 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))  # Φ(x)
    pdf_x = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)  # φ(x)
    return phi_x + x * pdf_x

def gelu_approx1(x: np.ndarray) -> np.ndarray:
    """Computes a GELU approximation based on tanh. Faster but less accurate."""
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))

def gelu_approx1_prime(x: np.ndarray) -> np.ndarray:
    """Computes a GELU derivative approximation based on tanh. Faster but less accurate."""
    c = np.sqrt(2 / np.pi)
    tanh_term = np.tanh(c * (x + 0.044715 * x**3))
    sech2_term = 1 - tanh_term**2
    return 0.5 * (1 + tanh_term) + 0.5 * x * sech2_term * c * (1 + 3 * 0.044715 * x**2)

def gelu_approx2(x: np.ndarray) -> np.ndarray:
    """Computes a GELU approximation based on sigmoid. Even faster but even less accurate too."""
    return x * (1 / (1 + np.exp(-1.702 * x)))

def gelu_approx2_prime(x: np.ndarray) -> np.ndarray:
    """Computes a GELU derivative approximation based on sigmoid. Even faster but even less accurate too."""
    sig = 1 / (1 + np.exp(-1.702 * x))
    return sig + 1.702 * x * sig * (1 - sig)

# SWISH (SiLU)
def swish(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))

def swish_prime(x: np.ndarray) -> np.ndarray:
    sig = 1 / (1 + np.exp(-x))
    return sig + x * sig * (1 - sig)

# Softmax
def softmax(x: np.ndarray) -> np.ndarray:
    """Applique la fonction softmax sur l'axe des classes (-1)."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stabilité numérique
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_prime_approx(x: np.ndarray) -> np.ndarray:
    """Approximation : considère que les éléments hors-diagonale sont négligeables"""
    s = softmax(x)
    return s * (1 - s)  # Approximation élément par élément (diag uniquement)

# Nécessite une modification particulière de la classe Network
# pour passer 2 paramètres en argument
# et ne pas utiliser le 2ème paramètre après.
def softmax_prime(x: np.ndarray, dL_dS: np.ndarray) -> np.ndarray:
    S = softmax(x)  # Obtenir S_i
    # Construire la matrice jacobienne complète pour chaque élément du batch
    batch_size, nb_classes = S.shape
    jacobian = np.einsum('bi,bj->bij', S, S)  # Produit externe : S_i * S_j
    np.einsum('bii->bi', jacobian)[...] = S * (1 - S)  # Remplace la diagonale par S_i (1 - S_i)
    # Appliquer la jacobienne au gradient dL/dS
    dL_dX = np.einsum('bij,bj->bi', jacobian, dL_dS)  # Multiplication jacobienne x gradient
    return dL_dX


__all__ = ["tanh", "tanh_prime", "relu", "relu_prime", "leakyrelu", "leakyrelu_prime", "gelu_approx2", "gelu_approx2_prime", "swish", "swish_prime", "softmax", "softmax_prime", "softmax_prime_approx"]
