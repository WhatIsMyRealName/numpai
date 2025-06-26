import numpy as np

# loss functions and there derivatives

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sum(np.power(y_true-y_pred, 2))
def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2*(y_pred-y_true) #/y_true.size

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    return -np.sum(y_true * np.log(y_pred))
def cross_entropy_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    # pas clair laquelle est la bonne...
    return - (y_true / y_pred)
    # return (y_pred - y_true) # si on met ça on doit "sauter" le softmax

def kl_divergence(y_true, y_pred, epsilon=1e-5):
    """Kullback-Leibler Divergence"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Évite log(0)
    mask = y_true > 0  # Stabilité numérique
    return np.sum(y_true[mask] * np.log(y_true[mask] / y_pred[mask])) # On a supprimé 0 * (+inf), qu'on a défini comme 0
def kl_divergence_prime(y_true, y_pred, epsilon=1e-12):
    """Dérivée de la divergence de Kullback-Leibler"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Éviter division par zéro
    # pas clair laquelle est la bonne...
    # return - (y_true / y_pred)
    return (y_pred - y_true)  

def hinge_loss(y_true, y_pred, delta=1.0):
    """Hinge Loss multi-classe"""
    correct_class_scores = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    margins = np.maximum(0, y_pred - correct_class_scores + delta) # Place la classe correcte à 1
    margins *= (1 - y_true)  # Ignore la classe correcte
    return np.sum(margins)
def hinge_loss_prime(y_true, y_pred):
    """Dérivée du Hinge Loss"""
    grad = np.where(1 - y_true * y_pred > 0, -y_true, 0)
    return grad

def categorical_hinge_loss(y_true, y_pred):
    """Calculates the categorical hinge loss between predicted and true labels.

    Args:
        y_true: True labels (one-hot encoded).
        y_pred: Predicted labels (probabilities).

    Returns:
        The sum of categorical hinge losses for all samples in the batch.
    """
    pos = np.sum(y_true * y_pred, axis=-1)
    neg = np.max((1. - y_true) * y_pred, axis=-1)
    # Compute loss for each sample in the batch
    loss = np.maximum(0., neg - pos + 1.)  
    # Return the sum of losses for all samples in the batch
    return np.sum(loss) # Return the total loss to maintain precision
def categorical_hinge_loss_prime(y_true, y_pred):
    """Dérivée de la Categorical Hinge Loss"""
    pos = y_true
    neg = np.max((1 - y_true) * y_pred, axis=-1, keepdims=True)
    grad = np.where(y_pred == pos, -1, 1)
    grad *= (1 + neg - y_pred > 0)
    return grad

def generalized_cross_entropy(y_true, y_pred, gamma=0.7, epsilon=1e-12):
    """Generalized Cross-Entropy Loss"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Évite log(0)
    return np.sum((1 - y_pred**gamma) / gamma * y_true)
def gce_loss_prime(y_true, y_pred, q=0.7, epsilon=1e-12):
    """Dérivée de la Generalized Cross-Entropy Loss"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Éviter log(0)
    return -q * (1 - y_pred**q)**(1 / (q - 1)) * y_true * y_pred**(q - 1)

def focal_loss(y_true, y_pred, gamma=2.0, epsilon=1e-12):
    """Focal Loss"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Évite log(0)
    return -np.sum((1 - y_pred) ** gamma * y_true * np.log(y_pred))
def focal_loss_prime(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-12):
    """Dérivée de la Focal Loss"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Éviter log(0)
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    return -alpha * y_true * (1 - pt) ** gamma * (gamma * y_pred * np.log(pt) + y_pred - 1)

def squared_hinge_loss(y_true, y_pred, delta=1.0):
    """Squared Hinge Loss"""
    correct_class_scores = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    margins = np.maximum(0, y_pred - correct_class_scores + delta) ** 2
    margins *= (1 - y_true)  # Ignore la classe correcte
    return np.sum(margins)
def squared_hinge_loss_prime(y_true, y_pred):
    """Dérivée du Squared Hinge Loss"""
    grad = np.where(1 - y_true * y_pred > 0, -2 * y_true * (1 - y_true * y_pred), 0)
    return grad

def js_divergence(y_true, y_pred, epsilon=1e-12):
    """Jensen-Shannon Divergence"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Évite log(0)
    y_true = np.clip(y_true, epsilon, 1)
    m = 0.5 * (y_true + y_pred)
    return 0.5 * (np.sum(y_true * np.log(y_true / m)) + np.sum(y_pred * np.log(y_pred / m)))
def js_divergence_prime(y_true, y_pred, epsilon=1e-12):
    """Dérivée de la divergence de Jensen-Shannon"""
    y_pred = np.clip(y_pred, epsilon, 1)  # Éviter division par zéro
    M = 0.5 * (y_true + y_pred)
    return 0.5 * (-y_true / M + -y_pred / M)

if __name__ == '__main__':
    # Exemple d'utilisation
    y_true = np.array([[0, 1, 0], [1, 0, 0]])  # Exemples de one-hot
    y_pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])  # Probabilités softmax
    print("MSE: ", mse(y_true, y_pred))
    print("Cross entropy loss: ", cross_entropy_loss(y_true, y_pred))
    print("KL Divergence:", kl_divergence(y_true, y_pred))
    print("Hinge Loss:", hinge_loss(y_true, y_pred))
    print("Categorical Hinge Loss:", categorical_hinge_loss(y_true, y_pred))
    print("Generalized Cross-Entropy:", generalized_cross_entropy(y_true, y_pred))
    print("Focal Loss:", focal_loss(y_true, y_pred))
    print("Squared Hinge Loss:", squared_hinge_loss(y_true, y_pred))
    print("JS Divergence:", js_divergence(y_true, y_pred))
