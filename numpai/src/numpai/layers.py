import numpy as np
from numpy.lib.stride_tricks import as_strided
import warnings

class Layer:
    """
    Classe de base de toutes les couches. Pas grand intérêt.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input, training=True):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    """
    Couche entrièrement connectée.

    Parameters
    ----------
    input_size : int
        Taille de l'entrée de la couche.
    output_size : int
        Taille de la sortie de la couche.
    init_method : str
        Méthode d'initialisation des poids et biais, par défaut "he".
        Choisir parmi "he", "xavier", ou "random".
        
    Methods
    ----------
    forward_propagation : Permet l'inférence.
    backward_propagation : Permet de mettre à jour la couche.
    
    Attributes
    ----------
    biases : np.ndarray
    weightst : np.ndarray
    input : np.ndarray
        Seulement après avoir exécuté forward_propagation.
    output : np.ndarray
        Idem.
        TODO : Supprimer cet attribut jamais utilisé
    """
    def __init__(self, input_size: int, output_size: int, init_method: str="he"):

        if init_method not in {"he", "xavier", "random"}:
            warnings.warn(f"""\033[93m\n[WARNING] init_method '{init_method}' inconnu. Utilisation de 'random' par défaut.\033[0m""", UserWarning)
            init_method = "random"

        self.biases = np.random.randn(1, output_size) * 0.1 # Introduit un bruit au départ
        # Initialisation des poids en fonction de la méthode choisie
        if init_method == "he":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.5 / input_size)
        elif init_method == "xavier":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.5 / input_size)
        else:  # Random
            self.weights = np.random.randn(input_size, output_size) * 0.1
            
        # print(f"Poids initialisés ({init_method}) : max={np.max(self.weights)}, min={np.min(self.weights)}, moy={np.mean(self.weights)}, std={np.std(self.weights)}")
        # print("Poids : ", self.weights)
        # print(f"Biais initialisés ({init_method}) : max={np.max(self.biases)}, min={np.min(self.biases)}, moy={np.mean(self.biases)}, std={np.std(self.biases)}")
        # print("Biais : ", self.biases)
        # Define m & v for weights and biases 
        # These variables are used in Adam optimization 
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_biases = np.zeros((1, output_size))
        self.v_biases = np.zeros((1, output_size))
        # Define hyperparameters for Adam optimizer 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 1

    def forward_propagation(self, input_data: np.ndarray, training: bool=True) -> np.ndarray:
        """
        Permet l'inférence des données passées en argument

        Parameters
        ----------
        input_data : np.ndarray
            Données à traiter.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Inutile pour ce type de couche (pas de différence entre True et False).

        Returns
        -------
        np.ndarray
            Sortie de la couche.
        """
        
        self.input = np.atleast_2d(input_data)  # Assure une entrée 2D même sans batch
        self.output = self.input @ self.weights + self.biases  # Produit matriciel optimisé
        return self.output if input_data.ndim > 1 else self.output.flatten()

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        """
        Permet la mise à jour des poids et biais de la couche.

        Parameters
        ----------
        output_error : np.ndarray
            L'erreur à la sortie de la couche.
        learning_rate : float
            Taux d'apprentissage de la couche.

        Returns
        -------
        np.ndarray
            Erreur à l'entrée de la couche.
        """
        output_error = np.atleast_2d(output_error)
        batch_size = self.input.shape[0]
        
        # Calcul des gradients
        d_weights = self.input.T @ output_error / batch_size
        d_biases = np.mean(output_error, axis=0, keepdims=True)
        
        # Clip pour éviter des mises à jour trop extrêmes
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)

        input_error = output_error @ self.weights.T

        # Mise à jour des poids avec Adam
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        # Mise à jour des biais avec Adam
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)
        m_hat_biases = self.m_biases / (1 - self.beta1 ** self.t)
        v_hat_biases = self.v_biases / (1 - self.beta2 ** self.t)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        # Mise à jour du compteur de temps Adam
        self.t += 1
        
        return input_error


class Conv2D(Layer):
    """
    Couche conovolutive.

    Parameters
    ----------
    input_shape : tuple
        Taille de l'entrée de la couche.
    num_filters : int
        Nombre de filtres de la couche.
    filter_size : tuple
        Taille des filtres.
    stride : int
        Pas des filtres pour la convolution, by default 1.
    padding : int
        Compensation des débordements des filtres lors de la convolution, by default 0.
    init_method : str
        Méthode d'initialisation des poids et biais, par défaut "he".
        Choisir parmi "he", "xavier", ou "random".
        
    Methods
    ----------
    forward_propagation : Permet l'inférence.
    backward_propagation : Permet de mettre à jour la couche.
    im2col : Fonction auxiliaire.
    
    Attributes
    ----------
    input_shape : tuple
    num_filters : int
    filter_size : tuple
    stride : int
    padding : int
    biases : np.ndarray
    padded_input : np.ndarray
    output : np.ndarray
        Seulement après avoir exécuté forward_propagation.
        TODO : Supprimer cet attribut jamais utilisé
    """
    def __init__(self, input_shape: tuple, num_filters: int, filter_size: tuple, stride: int=1, padding: int=0, init_method: str="he"):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape  # (channels, height, width)

        self.biases = np.random.uniform(-0.1, 0.1, size=(num_filters, 1, 1))
        if init_method not in {"he", "xavier", "random"}:
            warnings.warn(f"""\033[93m\n[WARNING] init_method '{init_method}' inconnu. Utilisation de 'random' par défaut.\033[0m""", UserWarning)
            init_method = "random"
        if init_method == "he":
            self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * np.sqrt(2 / (input_shape[0] * filter_size * filter_size))
        elif init_method == "xavier":
            self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * np.sqrt(1 / (input_shape[0] * filter_size * filter_size))
        else:  # "random"
            self.filters = np.random.uniform(-0.1, 0.1, size=(num_filters, input_shape[0], filter_size, filter_size))

        # Préallocation du buffer pour le padding (évite np.pad à chaque forward)
        padded_shape = (1, input_shape[0], input_shape[1] + 2 * padding, input_shape[2] + 2 * padding)
        self.padded_input = np.zeros(padded_shape)
        
    def im2col(self, input_data: np.ndarray) -> np.ndarray:
        """
        Fonction auxiliaire qui transforme l'entrée en une matrice colonne compatible avec une multiplication matricielle.
        Utilisée pour faire des convolutions rapides avec np.einsum.

        Parameters
        ----------
        input_data : np.ndarray
            Données à convertir, sous la forme [batch, channels, height, width]

        Returns
        -------
        np.ndarray
            Données sous la forme [batch, channels, height, width, filters_height, filters_width]
        """
        batch_size, channels, height, width = input_data.shape
        k, s = self.filter_size, self.stride

        out_height = (height - k) // s + 1
        out_width = (width - k) // s + 1

        strides = input_data.strides
        strided_input = as_strided(
            input_data,
            shape=(batch_size, channels, out_height, out_width, k, k),
            strides=(strides[0], strides[1], strides[2] * s, strides[3] * s, strides[2], strides[3]),
            writeable=False
        )

        return strided_input

    def forward_propagation(self, input_data: np.ndarray, training: bool=True) -> np.ndarray:
        """
        Permet l'inférence des données passées en argument.

        Parameters
        ----------
        input_data : np.ndarray
            Données à traiter.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Inutile pour ce type de couche (pas de différence entre True et False).

        Returns
        -------
        np.ndarray
            Sortie de la couche.
        """
        changed = False
        if input_data.ndim == 4:
          batch_size = input_data.shape[0]
        else:
          batch_size = 1
          changed = True

        # Gestion du padding (remplissage manuel pour éviter np.pad)
        if self.padded_input.shape[0] != batch_size:
            self.padded_input = np.zeros(
                (batch_size, self.input_shape[0], self.input_shape[1] + 2 * self.padding, self.input_shape[2] + 2 * self.padding)
            )
        self.padded_input[:, :, self.padding:-self.padding, self.padding:-self.padding] = input_data

        # Extraction des sous-matrices de convolution
        col_input = self.im2col(self.padded_input)

        # Calcul de la convolution en un seul appel
        output = np.einsum('bchwkl,dckl->bdhw', col_input, self.filters, optimize=True) + self.biases

        self.output = output
        return self.output[0] if changed else self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        """
        Permet la mise à jour des poids et biais de la couche.

        Parameters
        ----------
        output_error : np.ndarray
            Erreur à la sortie de la couche.
        learning_rate : float
            Taux d'apprentissage de la couche.

        Returns
        -------
        np.ndarray
            Erreur à l'entrée de la couche.
        """

        # Extraction des sous-matrices de l'entrée
        # déjà calculé lors de la porpagation avant... -> ajouter en attribut
        col_input = self.im2col(self.padded_input)

        # Calcul du gradient des filtres
        d_filters = np.einsum('bchwkl,bdhw->dckl', col_input, output_error, optimize=True)

        # Calcul du gradient du biais
        d_biases = np.sum(output_error, axis=(0, 2, 3), keepdims=True)

        # Calcul du gradient d'entrée
        # passer ça en attribut pour ne pas le recréer à chaque fois
        padded_error = np.zeros_like(self.padded_input)
        flipped_filters = np.flip(self.filters, axis=(2, 3))
        col_error = as_strided(padded_error, shape=col_input.shape, strides=col_input.strides, writeable=True)
        np.einsum('dckl,bdhw->bchwkl', flipped_filters, output_error, out=col_error, optimize=True)

        # Mise à jour des paramètres
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases.squeeze(axis=0)

        return padded_error[:, :, self.padding:-self.padding, self.padding:-self.padding]

class FlattenLayer(Layer):
    """
    Couche applatissante. Convertit toutes les données en vecteur 1D. Utile pour mettre des couches FC après des couches convolutives.
    
    Methods
    ----------
    forward_propagation : Permet l'applatissement des données.
    backward_propagation : Permet de reconsituer l'erreur sous la forme d'entrée.
    
    Attributes
    ----------
    input_shape : tuple
        Seulement après forward_propagation
    """
    def __init__(self):
        self.input_shape = None  # Stocke la forme d'origine pour la backpropagation

    def forward_propagation(self, input_data : np.ndarray, training: bool=True) -> np.ndarray:
        """
        Permet l'applatissement des données.

        Parameters
        ----------
        input_data : np.ndarray
            Données à applatir.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Inutile pour ce type de couche (pas de différence entre True et False).

        Returns
        -------
        np.ndarray
            Données applaties.

        Raises
        ------
        ValueError
            Si les données reçues n'ont pas le nombre de dimensions attendues.
            TODO : Supprimer cette erreur inutile.
        """
        self.input_shape = input_data.shape  # Sauvegarde la forme originale

        # Vérifie si input_data a un batch ou non
        if input_data.ndim == 3:  # Une seule image (C, H, W)
            return input_data.flatten()
        elif input_data.ndim == 4:  # Batch d'images (N, C, H, W)
            return input_data.reshape(input_data.shape[0], -1)
        else:
            raise ValueError(f"FlattenLayer a reçu une entrée avec {input_data.ndim} dimensions, attendu 3 ou 4.")

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        """
        Permet de reconsituer l'erreur sous la forme d'entrée.

        Parameters
        ----------
        output_error : np.ndarray
            Erreur à la sortie de la couche.
        learning_rate : float, optional
            Taux d'apprentissage de la couche.
            Inutile pour ce type de couche (pas de différence quelle que soit la valeur).

        Returns
        -------
        np.ndarray
            Erreur à l'entrée de la couche.
            NB : C'est la même erreur qu'à la sortie, avec des dimensions différentes.
        """
        return output_error.reshape(self.input_shape)  # Restaure la forme originale

class ActivationLayer(Layer):
    """
    Couche d'activation pour la non-linéarité du réseau.

    Parameters
    ----------
    activation 
        La fonction d'activation.
    activation_prime 
        La dérivée de la fonction d'activation.
    
    Methods
    ----------
    forward_propagation : Calcule l'image des données par la fonction d'activation.
    backward_propagation : Calcule l'erreur d'entrée.
    
    Attributes
    ----------
    activation 
    activation_prime 
    """
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data: np.ndarray, training: bool=True):
        """
        Calcule l'image des données par la fonction d'activation

        Parameters
        ----------
        input_data : np.ndarray
            Entrée de la couche.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Inutile pour ce type de couche (pas de différence entre True et False).

        Returns
        -------
        np.ndarray
            Image de l'entrée.
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int):
        """
        Calcule l'erreur à l'entrée de la couche.

        Parameters
        ----------
        output_error : np.ndarray
            Erreur à le sortie de la couche.
        learning_rate : float
            Taux d'apprentissage.

        Returns
        -------
        np.ndarray
            Erreur à l'entrée.
        """
        return self.activation_prime(self.input) * output_error

class Dropout(Layer):
    """
    Couche qui désactive aléatoirement certains neurones pour éviter le surapprentissage.

    Parameters
    ----------
    dropout_rate : float
        Taux de neurones désactivés.
    """
    def __init__(self, dropout_rate: float):
        assert 0 <= dropout_rate <= 1
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_propagation(self,  input_data: np.ndarray, training: bool=True) -> np.ndarray:
        """
        Met aléatoirement certaines données d'entrée à 0 et compense avec les autres.

        Parameters
        ----------
        input_data : np.ndarray
            Entrée de la couche.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Si False, cette fonction ne fait rien car on garde tous les neurones en inférence réelle (test ou utilisation).

        Returns
        -------
        np.ndarray
            Sortie de la couche, identique à l'entrée si `training=False`.
        """
        if training:
            # Générer le masque pour le dropout
            if self.mask is None:
                self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape)
            return input_data * self.mask / (1 - self.dropout_rate)
        return input_data

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        """
        Calcule l'erreur à l'entrée de la couche.

        Parameters
        ----------
        output_error : np.ndarray
            Erreur à la sortie de la couche.
        learning_rate : float
            Taux d'apprentissage de la couche.

        Returns
        -------
        np.ndarray
            Erreur à l'entrée de la couche. Pas d'erreur calculée sur les neurones désactivés.
        """
        if self.mask is not None:
            output_error *= self.mask
            # voir ici s'il faut * ou /
            output_error *= (1 - self.dropout_rate)
            self.mask = None
            return output_error
        return output_error

class MaxPooling(Layer):
    """
    Couche qui permet de réduire les dimensions spatiales des données 2D.

    Parameters
    ----------
    pool_size : tuple
        Taille de la zone dont on extrait le maximum.
    stride : int
        Pas entre chaque zone.
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data: np.ndarray, training: bool=True) -> np.ndarray:
        """
        Réduit les dimensions spatiales des données en ne gardant que le maximum de chaque zone.

        Parameters
        ----------
        input_data : np.ndarray
            Données d'entrée.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Inutile pour ce type de couche (pas de différence entre True et False).

        Returns
        -------
        np.ndarray
            Données réduites.
        """
        changed = False
        if input_data.ndim == 3:  # Cas sans batch, on ajoute une dimension batch
            input_data = np.expand_dims(input_data, axis=0)
            changed = True

        self.input = input_data  # Stockage pour backward
        batch_size, channels, height, width = input_data.shape
        pool_h, pool_w, stride = self.pool_size, self.pool_size, self.stride

        # Calcul de la taille de sortie
        output_height = (height - pool_h) // stride + 1
        output_width = (width - pool_w) // stride + 1

        # Découpage en blocs pour éviter les boucles (as_strided)
        shape = (batch_size, channels, output_height, output_width, pool_h, pool_w)
        strides = input_data.strides[:2] + (stride * input_data.strides[2], stride * input_data.strides[3]) + input_data.strides[2:]
        windows = np.lib.stride_tricks.as_strided(input_data, shape=shape, strides=strides)

        # Max pooling : trouver le maximum dans chaque région
        self.max_indices = np.argmax(windows.reshape(batch_size, channels, output_height, output_width, -1), axis=-1)
        self.output = np.max(windows, axis=(-2, -1))

        return self.output[0] if changed else self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        """
        Reconstitue l'erreur à l'entrée de la couche.

        Parameters
        ----------
        output_error : np.ndarray
            Erreur à la sortie de la couche.
        learning_rate : float
            Taux d'apprentissage de la couche.

        Returns
        -------
        np.ndarray
            Erreur à l'entrée de la couche.
        """
        if output_error.ndim == 3:  # Cas sans batch
            output_error = np.expand_dims(output_error, axis=0)

        input_error = np.zeros_like(self.input)  # Initialisation de l'erreur d'entrée

        batch_size, channels, output_height, output_width = output_error.shape
        pool_h, pool_w, stride = self.pool_size, self.pool_size, self.stride

        # Reconstruction des indices absolus du maximum sélectionné
        max_i, max_j = np.unravel_index(self.max_indices, (pool_h, pool_w))

        # Générer les indices batch et canal
        batch_idx = np.arange(batch_size)[:, None, None, None]
        channel_idx = np.arange(channels)[None, :, None, None]
        out_y = np.arange(output_height)[None, None, :, None] * stride
        out_x = np.arange(output_width)[None, None, None, :] * stride

        # Transformer en indices absolus
        abs_max_i = out_y + max_i
        abs_max_j = out_x + max_j

        # Propagation de l'erreur aux positions des maximums
        input_error[batch_idx, channel_idx, abs_max_i, abs_max_j] = output_error

        return input_error[0] if batch_size == 1 else input_error
        
class GlobalAvgPool2D(Layer):
    """
    Global Average Pooling 2D layer.

    This layer computes the average of each feature map (channel) over its spatial dimensions.

    Attributes
    ----------
    input_shape : tuple
        Shape of the input stored during forward propagation. Format: (batch_size, channels, height, width) or (channels, height, width).
    """
    def __init__(self):
        """
        Initialize the GlobalAvgPool2D layer.

        No parameters to set for this layer.
        """
        super().__init__()

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform the forward pass of global average pooling.

        Each channel is averaged over its height and width dimensions.

        Parameters
        ----------
        input_data : np.ndarray
            Input tensor of shape:
            - (batch_size, channels, height, width) when using mini-batches, or
            - (channels, height, width) for single samples without batch dimension.
        training : bool, optional
            Ignored for this layer (global average pooling has no train-time behavior differences), by default True.

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch_size, channels) or (channels,),
            containing the spatial averages for each channel.
        """
        changed = False
        if input_data.ndim == 3:
            # Add batch dimension if missing
            input_data = input_data[np.newaxis, ...]
            changed = True
        # Store input shape for backward pass
        self.input_shape = input_data.shape
        batch_size, channels, height, width = input_data.shape
        # Compute mean over spatial dimensions H and W
        self.output = input_data.mean(axis=(2, 3))  # shape: (batch_size, channels)
        # Remove added batch dimension if necessary
        return self.output[0] if changed else self.output

    def backward_propagation(self,
                             output_error: np.ndarray,
                             learning_rate: float,
                             epoch: int) -> np.ndarray:
        """
        Perform the backward pass of global average pooling.

        The gradient is distributed equally across all spatial locations.

        Parameters
        ----------
        output_error : np.ndarray
            Gradient of the loss with respect to the layer output,
            of shape (batch_size, channels) or (channels,).
        learning_rate : float
            Learning rate (not used in this layer).
        epoch : int
            Current epoch index (not used in this layer).

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the layer input,
            of shape matching the original input: (batch_size, channels, height, width) or (channels, height, width).
        """
        # Ensure output_error has batch dimension
        if output_error.ndim == 1:
            output_error = output_error[np.newaxis, ...]
        batch_size, channels = output_error.shape
        _, _, height, width = self.input_shape
        # Distribute gradient uniformly over H*W spatial positions
        grad = output_error[:, :, None, None] / (height * width)
        # Broadcast to full input shape
        input_error = np.ones(self.input_shape) * grad
        # Remove batch dimension if original input had none
        return input_error[0] if batch_size == 1 else input_error


class BatchNormalization:
    """
    Couche pour normaliser les données, pour réduire le surapprentissage et accélérer la convergence du réseau.

    Parameters
    -------
    input_size : tuple
        Taille des données d'entrée.
    momentum : float
        Coefficient pour lisser les normalisations d'une fois sur l'autre, by default 0.9.
    epsilon : float
        Petite valeur pour la stabilité numérique, by default 1e-5.

    Raises
    ------
    ValueError
        Si input_size n'est pas un entier ni un tuple de dimension 3.
    ValueError
        Si la dimension de l'entrée est strictement plus grande que 4.
    """
    # Note that the vocabulary used in this class is different than the one used in other layers
    def __init__(self, input_size: tuple, momentum: float=0.9, epsilon: float=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_size = input_size

        if isinstance(input_size, int):  
            # Fully Connected (input_size = (x,))
            feature_shape = (1, input_size)  
        elif len(input_size) == 3:  
            # Convolutionnel (input_size = (c, h, w))
            feature_shape = (1, input_size[0], *([1] * (len(input_size) - 1)))
        else:
            raise ValueError(f"BatchNormalization ne supporte pas input_size={input_size} (dims n'est pas 1 ou 3)")

        # Initialisation des paramètres
        self.gamma = np.ones(feature_shape)
        self.beta = np.zeros(feature_shape)

        self.running_mean = np.zeros(feature_shape)
        self.running_var = np.ones(feature_shape)

    def forward_propagation(self, input_data: np.ndarray, training: bool=True) -> np.ndarray:
        """
        Permet de normaliser les données. Fonctionne même avec les entrées sans batch.

        Parameters
        ----------
        input_data : np.ndarray
            Entrée de la couche.
        training : bool, optional
            Si l'IA est en train de s'entraîner ou pas, by default True.
            Inutile pour ce type de couche (pas de différence entre True et False).

        Returns
        -------
        np.ndarray
            Données normalisées.

        Raises
        ------
        ValueError
            Si la dimension de l'entrée est strictement supérieure à 4.
        """
        input_ndim = input_data.ndim

        if input_ndim > 4:
            raise ValueError(f"BatchNormalization ne supporte pas input_data.ndim={input_ndim} (> 4).")

        # Détection du format (FC ou Conv)
        if input_ndim == 2:
            reduce_axes = (0,)  # FC : normalisation sur le batch
        elif input_ndim == 4:
            reduce_axes = (0, 2, 3)  # Conv : normalisation sur batch + H + W
        else:
            return input_data  # Pas de normalisation si 1D ou 3D

        batch_size = input_data.shape[0]

        # Vérification du batch_size pour décider si on normalise ou non
        if training and batch_size > 1:  
            mean = np.mean(input_data, axis=reduce_axes, keepdims=True)
            variance = np.var(input_data, axis=reduce_axes, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * variance
        else:
            mean = self.running_mean
            variance = self.running_var

        self.input = input_data
        self.mean = mean
        self.variance = variance
        self.normalized_input = (input_data - mean) / np.sqrt(variance + self.epsilon)
        
        return self.gamma * self.normalized_input + self.beta

    def backward_propagation(self, dout: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        """
        Calcule l'erreur à l'entrée de la couche.

        Parameters
        ----------
        dout : np.ndarray
            Erreur à la sortie de la couche.
        learning_rate : float
            Taux d'apprentissage de la couche.

        Returns
        -------
        np.ndarray
            Erreur à l'entrée de la couche (i.e. erreur non normalisée).
        """
        input_ndim = self.input.ndim
        batch_size = self.input.shape[0] if input_ndim in [2, 4] else 1  

        # Si pas de normalisation à l'aller, rien à faire ici non plus
        if input_ndim in {1, 3} or batch_size == 1:
            return dout * self.gamma

        if input_ndim == 2:
            reduce_axes = (0,)  
        elif input_ndim == 4:
            reduce_axes = (0, 2, 3)  

        dgamma = np.sum(dout * self.normalized_input, axis=reduce_axes, keepdims=True)
        dbeta = np.sum(dout, axis=reduce_axes, keepdims=True)

        dx_hat = dout * self.gamma
        N = np.prod([self.input.shape[i] for i in reduce_axes])  

        dx_centered1 = dx_hat / np.sqrt(self.variance + self.epsilon)
        dvar = np.sum(dx_hat * (self.input - self.mean) * -0.5 * (self.variance + self.epsilon) ** -1.5, axis=reduce_axes, keepdims=True)
        dx_centered2 = 2 * (self.input - self.mean) * dvar / N
        dmean = np.sum(-(dx_centered1 + dx_centered2), axis=reduce_axes, keepdims=True)
        dx = dx_centered1 + dx_centered2 + dmean / N

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dx



# TODO : Document the end of this file
# Embedding Layer (for integer input indices)
class Embedding(Layer):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = np.random.randn(vocab_size, embed_dim)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.embeddings[input_data]
        return self.output

    def backward_propagation(self, output_error, learning_rate, epoch: int):
        for i, index in enumerate(self.input):
            self.embeddings[index] -= learning_rate * output_error[i]
        return None  # No input error needed for embedding layer

# Recurrent Layers - Simplified LSTM
class LSTM(Layer):
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Weights for input, forget, output and candidate gates
        self.W = np.random.randn(4, input_dim + hidden_dim, hidden_dim)
        self.b = np.random.randn(4, hidden_dim)
        
    def forward_propagation(self, input_data):
        self.h, self.c = np.zeros(self.hidden_dim), np.zeros(self.hidden_dim)
        self.outputs = []
        for t in range(input_data.shape[1]):  # Sequence length
            x_t = input_data[:, t, :]
            combined = np.concatenate((x_t, self.h), axis=1)
            i_t = self._sigmoid(np.dot(combined, self.W[0]) + self.b[0])  # Input gate
            f_t = self._sigmoid(np.dot(combined, self.W[1]) + self.b[1])  # Forget gate
            o_t = self._sigmoid(np.dot(combined, self.W[2]) + self.b[2])  # Output gate
            g_t = np.tanh(np.dot(combined, self.W[3]) + self.b[3])        # Candidate
            self.c = f_t * self.c + i_t * g_t
            self.h = o_t * np.tanh(self.c)
            self.outputs.append(self.h)
        return np.stack(self.outputs, axis=1)

    def backward_propagation(self, output_error, learning_rate, epoch: int):
        raise NotImplementedError
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


__all__ = [FCLayer, Conv2D, FlattenLayer, ActivationLayer, Dropout, BatchNormalization, MaxPooling]