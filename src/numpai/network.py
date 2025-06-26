from __future__ import annotations # type hinting
# To load/save model
import pickle

# For fit2() and fit3()
import os
# This is for display only
import sys
from math import ceil

# For summary() and fit()
# For deprecation warnings (fit)
import warnings
# For summary() and export() (export not implemented yet)
import matplotlib.pyplot as plt
from layers import Layer, FCLayer, Conv2D, BatchNormalization, Dropout, FlattenLayer, MaxPooling, ActivationLayer, GlobalAvgPool2D

# For type hinting (not necessary)
import numpy as np # Note that `numpy` is already imported by `layer` and that you realy need it.
from typing import Type
from pathlib import Path

# TODO : use try/except for optionals dependancies and allow methods to run without

class Network:
    """
    Classe représentant un réseau de neurones.
        
    Methods
    ----------
    add : Ajoute une couche au réseau.
    use : Définit la fonction d'erreur.
    predict : Faire une prédiction.
    fit - deprecated : Entraîne le réseau.
    fit2 : Entraîne le réseau.
    summary : Résumé du réseau.
    save : Enregistre le réseau.
    load - staticmethod : Charge un réseau.
    
    Attributes
    ----------
    layers : list
    loss : function
    loss_prime : function
    logs : list
    training_data : dict
    """
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.logs = []  # erreur
        self.logs_test = []   # erreur de validation
        self.training_data = {"X": [], "y": []} # pour stocker des données d'entraînement lors de son utilisation dans un autre programme

    def add(self, layer: Type[Layer]) -> None:
        self.layers.append(layer)

    def use(self, loss: function, loss_prime: function) -> None:
        """
        Permet de définir la fonction d'erreur utilisée par le réseau.

        Parameters
        ----------
        loss : function
            Fonction d'erreur.
        loss_prime : function
            Dérivée de la fonction d'erreur.
        """
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data: list|np.ndarray) -> list:
        """
        Permet l'inférence du modèle sur les données passées en argument.

        Parameters
        ----------
        input_data : list | np.ndarray
            Liste des objets sur lesquels inférer. Eventuellement une liste à 1 élément.

        Returns
        -------
        list
            liste des résultats pour chaque objet d'entrée.
        """
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output, training=False)
            result.append(output)

        return result

    def fit(self, x_train: list|np.ndarray, y_train: list|np.ndarray, epochs: int, learning_rate: float) -> None:
        
        warnings.warn(f"""\033[93m\n[WARNING] 'fit' est déprécié et pourrait être supprimé dans les prochaines versions. Utilisez fit2 à la place.\033[0m""", DeprecationWarning)

        samples = len(x_train)
        initial_learning_rate = learning_rate
        decay = 0.001
        for i in range(epochs):
            err = 0
            for j in range(samples):
                sys.stdout.write(f"\rforward propagation... {j+1}/{samples}")
                sys.stdout.flush()
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                sys.stdout.write(f"\backward propagation... {j+1}/{samples}   ")
                sys.stdout.flush()
                learning_rate = initial_learning_rate / (1 + decay * i)
                error = self.loss_prime(y_train[j], output)
                # print(output, y_train[j], error)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate, i)

            # calculate average error on all samples
            err /= samples
            sys.stdout.write('\r')
            print('epoch %d/%d              error=%f' % (i+1, epochs, err))

    def fit2(self, x_train: list|np.ndarray, y_train: list|np.ndarray, epochs: int, learning_rate: float, batch_size: int=32) -> None:
        """
        Permet d'entraîner le modèle

        Parameters
        ----------
        x_train : list | np.ndarray
            Liste des données d'entraînement.
        y_train : list | np.ndarray
            Liste des sorties attendues pour chaque entrée.
        epochs : int
            Nombre d'itérations sur les données.
        learning_rate : float
            Taux d'apprentissage du modèle.
        batch_size : int, optional
            Nombre de données traitées d'un coup, by default 32.
        """
        samples = len(x_train)
        initial_learning_rate = learning_rate
        decay = 0.001
        for i in range(epochs):
            err = 0.
            for j in range(0, samples, batch_size):
                # Sélection du mini-lot
                x_batch = x_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]
                sys.stdout.write(f"\rforward propagation... {j/batch_size +1}/{ceil(samples/batch_size)}   ")
                sys.stdout.flush()

                # Propagation avant
                output = x_batch
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Calcul de la perte pour le mini-lot
                # batch_loss = [self.loss(y_batch[k], output[k]) for k in range(len(y_batch))]
                batch_loss = self.loss(y_batch, output)
                err += batch_loss

                # Rétropropagation pour le mini-lot
                learning_rate = initial_learning_rate # / (1 + decay * i) # faut voir, c'est déjà pris en compte par Adam ou pas ?
                sys.stdout.write(f"\rbackward propagation... {j/batch_size +1}/{ceil(samples/batch_size)}")
                sys.stdout.flush()
                # batch_error = np.array([self.loss_prime(y_batch[k], output[k]) for k in range(len(y_batch))])
                batch_error = self.loss_prime(y_batch, output)
                # print(output, y_batch, batch_error)
                for layer in reversed(self.layers):
                    batch_error = layer.backward_propagation(batch_error, learning_rate, i)

            # Calcul de la perte moyenne pour cette époque
            err /= samples
            sys.stdout.write('\r')
            print('epoch %d/%d                    error=%f' % (i+1, epochs, err))
            self.logs.append(err)

    def fit3(self,
            x_train: list|np.ndarray,
            y_train: list|np.ndarray,
            x_val: list|np.ndarray,
            y_val: list|np.ndarray,
            epochs: int,
            learning_rate: float,
            batch_size: int = 32,
            patience_lr_decay: int = 2,
            patience_early_stopping: int = 5,
            min_lr: float = 1e-5) -> None:
        """
        Entraîne le modèle avec validation et early stopping.

        Parameters
        ----------
        x_train : list | np.ndarray
            Données d'entraînement.
        y_train : list | np.ndarray
            Étiquettes d'entraînement.
        x_val : list | np.ndarray
            Données de validation.
        y_val : list | np.ndarray
            Étiquettes de validation.
        epochs : int
            Nombre d'époques d'entraînement.
        learning_rate : float
            Taux d'apprentissage initial.
        batch_size : int, optional
            Taille des mini-lots, par défaut 32.
        patience : int, optional
            Nombre d'époques sans amélioration avant arrêt, par défaut 5.

        Return
        --------
        Network
            Meilleur modèle lors de l'entraînement
        """
        samples = len(x_train)
        decay = 0.001

        best_val_err = float('inf')
        wait_stop = 0
        wait_lr = 0

        for epoch in range(1, epochs + 1):
            epoch_err = 0.0

            # --- Phase d'entraînement ---
            for j in range(0, samples, batch_size):
                x_batch = x_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]
                sys.stdout.write(f"\rforward propagation... {int(j/batch_size) +1}/{ceil(samples/batch_size)}   ")
                sys.stdout.flush()

                # Propagation avant
                output = x_batch
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Calcul de la perte
                batch_loss = self.loss(y_batch, output)
                epoch_err += batch_loss

                sys.stdout.write(f"\rbackward propagation... {int(j/batch_size) +1}/{ceil(samples/batch_size)}")
                sys.stdout.flush()
                # Rétropropagation
                # lr = learning_rate / (1 + decay * epoch)
                error_batch = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    error_batch = layer.backward_propagation(error_batch, learning_rate, epoch - 1)

            # Moyenne de la perte d'entraînement
            train_err = epoch_err / samples
            self.logs.append(train_err)

            # --- Phase de validation ---
            # Propagation avant sur données de validation
            val_output = self.predict(x_val)
            val_err = self.loss(y_val, val_output) / len(x_val)
            self.logs_test.append(val_err)

            # Affichage
            sys.stdout.write('\r')
            print(f"Epoch {epoch}/{epochs} - train_error={train_err:.6f} - val_error={val_err:.6f}")


            # Early stopping et sauvegarde du meilleur modèle
            if val_err < best_val_err:
                best_val_err = val_err
                wait_stop = 0
                wait_lr = 0
                self.save("_tmp_model_file.pkl", debug=False)
            else:
                wait_stop += 1
                wait_lr += 1
                # Scheduler LR
                if wait_lr >= patience_lr_decay and learning_rate > min_lr:
                    learning_rate = max(learning_rate * 0.5, min_lr)
                    print(f"Nouveau learning rate : {learning_rate:.1e}")
                    wait_lr = 0
                # Early stopping
                if wait_stop >= patience_early_stopping:
                    print(f"Arrêt précoce après {epoch} époques (patience={patience_early_stopping}).")
                    break

        # Charger le meilleur modèle
        best_model = self.__class__.load("_tmp_model_file.pkl", debug=False)
        os.remove("_tmp_model_file.pkl")
        # Remplacer uniquement les poids/layers pour conserver logs complets
        self.layers = best_model.layers

    def summary(self) -> None:
        """
        Affiche un résumé du modèle.
        """
        print("\n\nModel Summary:\n")
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__

            # Calcul de la forme de sortie pour les couches Conv2D
            if isinstance(layer, Conv2D):
                # Extraction des paramètres de la couche Conv2D
                input_height, input_width = layer.input_shape[1], layer.input_shape[2]
                filter_height, filter_width = layer.filters.shape[2], layer.filters.shape[3]
                stride = layer.stride
                padding = layer.padding  # Padding est un entier
                
                # Calcul de la taille de sortie pour chaque dimension
                output_height = (input_height - filter_height + 2 * padding) // stride + 1
                output_width = (input_width - filter_width + 2 * padding) // stride + 1
                
                output_shape = (layer.num_filters, output_height, output_width)
            
            elif isinstance(layer, FCLayer):
                output_shape = (layer.biases.shape[1],)
            else:
                output_shape = getattr(layer, 'output_size', None)
                if output_shape is None:
                   output_shape = getattr(layer, 'input_size', None)
                   if output_shape is None:
                    output_shape = "N/A"

            # Compte des paramètres pour la couche actuelle
            params_count = 0

            # Pour les couches Conv2D
            if isinstance(layer, Conv2D):
                filter_size = layer.filters.shape[2] * layer.filters.shape[3]  # filtre (height * width)
                in_channels = layer.input_shape[0]  # canaux d'entrée
                params_count += (filter_size * in_channels + 1) * layer.num_filters  # (filtres * in_channels + biais)

            # Pour les couches Fully Connected (Dense)
            if hasattr(layer, 'weights'):
                params_count += layer.weights.size  # Nombre d'éléments dans les poids
            if hasattr(layer, 'bias'):
                params_count += layer.bias.size  # Nombre d'éléments dans les biais
            
            # Pour les couches BatchNormalization
            if isinstance(layer, BatchNormalization):
                params_count += 2 * layer.input_size[0] if isinstance(layer.input_size, tuple) else 2 * layer.input_size

            total_params += params_count

            print(f"Layer {i + 1}: {layer_type} | Output Shape: {output_shape} | Parameters: {params_count}")

        print(f"Total Parameters: {total_params}")
        if self.logs:
            plt.plot(self.logs, marker='o', linestyle='-', color='b', label='Erreur')
            plt.plot(self.logs_test, marker='o', linestyle='-', color='r', label='Erreur de validation')
            plt.title("Erreur au cours du temps (epochs)", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Erreur (UA)", fontsize=12)
            plt.grid(True)
            plt.legend(loc="best")
            plt.show()
    
    # Save the model to a file
    def save(self, filename: str|Path, debug=True) -> None:
        """
        Permet d'enregistrer le modèle.
        NOTE : On peut toujours entraîner un modèle qui a été sauvegardé par cette méthode.

        Parameters
        ----------
        filename : str | Path
            Nom du fichier où sera stocké le modèle. Peut contenir un chemin absolu ou relatif.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        if debug: print(f"Model saved to {filename}")

    # Load the model from a file
    @staticmethod
    def load(filename: str|Path, debug=True) -> Network:
        """
        Permet de charger un modèle

        Parameters
        ----------
        filename : str | Path
            Nom du fichier où est stocké le modèle.

        Returns
        -------
        Network
            Le modèle.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        if debug: print(f"Model loaded from {filename}")
        return model









    # En développement
    def export(self) -> None:
        """
        Export optimized model for inference.

        Removes training-specific layers and parameters, and fuses
        BatchNormalization into preceding layers for faster inference.

        Parameters
        ----------
        self : Network
            Neural network instance to be exported for inference.

        Returns
        -------
        None

        Notes
        -----
        - Dropout layers are removed entirely.
        - BatchNormalization parameters (gamma, beta, running mean/var)
          are fused into preceding Conv2D or FCLayer weights and biases:

            Given input x to BN layer:
                y = gamma * (x - mean) / sqrt(var + eps) + beta

            For a linear layer z = W x + b, substituting x from BN:
                z = W [(x - mean) / sqrt(var + eps) * gamma] + W[beta] + b

            So:
            - W_new = W * (gamma / sqrt(var + eps))
            - b_new = (b - mean) * (gamma / sqrt(var + eps)) + beta

        - Training-only methods (`fit`, `fit2`, `fit3`) and attributes
          (`loss`, `loss_prime`, `logs`, `logs_test`, `training_data`) are deleted.
        - Optimizer moment parameters in FCLayer (Adam: m_weights, v_weights,
          m_biases, v_biases, beta1, beta2, epsilon, t) are removed.
        - Sets `_inference_only` to True
          TODO: ensure `_inference_only` is initialized to False in `__init__`.
        """
        fused_layers = []
        for layer in self.layers:
            # Remove Dropout layers
            if isinstance(layer, Dropout):
                continue

            # Fuse BatchNormalization into previous layer
            if isinstance(layer, BatchNormalization) and fused_layers:
                prev = fused_layers[-1]

                # BN params
                gamma = layer.gamma            # scale parameter
                beta = layer.beta              # shift parameter
                mean = getattr(layer, 'mean', layer.running_mean)
                var = getattr(layer, 'variance', layer.running_var)
                eps = layer.epsilon

                # Compute scale = gamma / sqrt(var + eps)
                if isinstance(prev, FCLayer):
                    scale = gamma.flatten() / np.sqrt(var.flatten() + eps)
                    prev.weights *= scale[np.newaxis, :]
                    prev.biases = (prev.biases - mean.flatten()) * scale + beta.flatten()

                elif isinstance(prev, Conv2D):
                    scale = gamma.reshape(prev.filters.shape[0], 1, 1, 1) / \
                            np.sqrt(var.reshape(prev.filters.shape[0], 1, 1, 1) + eps)
                    prev.filters *= scale
                    prev.biases = (prev.biases - mean.reshape(prev.biases.shape)) * \
                                  scale.reshape(prev.biases.shape) + beta.reshape(prev.biases.shape)

                continue

            # Strip Adam optimizer parameters from FCLayer (training only)
            if isinstance(layer, FCLayer):
                for attr in ['m_weights', 'v_weights', 'm_biases', 'v_biases',
                            'beta1', 'beta2', 'epsilon', 't']:
                    if hasattr(layer, attr):
                        delattr(layer, attr)

            fused_layers.append(layer)

        # Assign optimized layers
        self.layers = fused_layers

        # Delete training methods
        for method in ['fit', 'fit2', 'fit3']:
            if hasattr(self, method):
                delattr(self, method)

        # Delete training attributes
        for attr in ['loss', 'loss_prime', 'logs', 'logs_test', 'training_data']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Mark as inference-only
        self._inference_only = True

    def export_to_onnx(self, filepath: str, input_shape: tuple) -> None:
        """
        Export the inference-optimized network to ONNX format.

        Parameters
        ----------
        filepath : str
            Path to save the .onnx model file. Utilisé par `onnx.save`.
        input_shape : tuple of int
            Shape of the input tensor, e.g., (batch_size, channels, height, width). 
            Passé à `helper.make_tensor_value_info` pour définir la signature.

        Returns
        -------
        None

        Notes
        -----
        - Conv2D -> Conv
        - FCLayer -> Gemm
        - ActivationLayer -> op (Relu, Tanh, Sigmoid, etc.) via mapping explicite
        """
        import onnx
        from onnx import helper, TensorProto
        nodes = []
        initializers = []

        # 1) Définition de l'entrée
        # helper.make_tensor_value_info(name, elem_type, shape)
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_shape)
        )
        prev_output = 'input'

        # Mapping des activations Python vers ONNX
        act_map = {
            'relu': 'Relu',
            'tanh': 'Tanh',
            'sigmoid': 'Sigmoid',
            'leakyrelu': 'LeakyRelu'
            # ajouter d'autres si nécessaire
        }

        # 2) Parcours des couches optimisées
        for idx, layer in enumerate(self.layers):
            # 2.a Conv2D -> Conv
            if isinstance(layer, Conv2D):
                W_name = f'conv{idx}_W'
                B_name = f'conv{idx}_B'
                # Crée et ajoute les tenseurs de poids et biais
                initializers.append(helper.make_tensor(
                    W_name, TensorProto.FLOAT,
                    list(layer.filters.shape),
                    layer.filters.flatten().tolist()
                ))
                initializers.append(helper.make_tensor(
                    B_name, TensorProto.FLOAT,
                    list(layer.biases.shape),
                    layer.biases.flatten().tolist()
                ))
                out_name = f'conv{idx}_out'
                # Crée le noeud Conv
                nodes.append(helper.make_node(
                    'Conv',
                    inputs=[prev_output, W_name, B_name],
                    outputs=[out_name],
                    name=f'Conv_{idx}',
                    pads=[layer.padding]*4 if hasattr(layer, 'padding') else [0,0,0,0],
                    strides=[layer.stride, layer.stride] if hasattr(layer, 'stride') else [1,1]
                ))
                prev_output = out_name

            # 2.b FCLayer -> Gemm
            elif isinstance(layer, FCLayer):
                W_name = f'fc{idx}_W'
                B_name = f'fc{idx}_B'
                initializers.append(helper.make_tensor(
                    W_name, TensorProto.FLOAT,
                    list(layer.weights.shape),
                    layer.weights.flatten().tolist()
                ))
                initializers.append(helper.make_tensor(
                    B_name, TensorProto.FLOAT,
                    list(layer.biases.shape),
                    layer.biases.flatten().tolist()
                ))
                out_name = f'fc{idx}_out'
                # Gemm: y = alpha * A * B + beta * C (alpha=1, beta=1 par défaut)
                nodes.append(helper.make_node(
                    'Gemm',
                    inputs=[prev_output, W_name, B_name],
                    outputs=[out_name],
                    name=f'Gemm_{idx}'
                ))
                prev_output = out_name

            # 2.c ActivationLayer (séparé)
            elif isinstance(layer, ActivationLayer):
                act_fn = layer.activation.__name__.lower()
                onnx_op = act_map.get(act_fn)
                if onnx_op is None:
                    raise ValueError(f"Activation '{act_fn}' non supportée pour ONNX export")
                act_name = f'act{idx}'
                nodes.append(helper.make_node(
                    onnx_op,
                    inputs=[prev_output],
                    outputs=[act_name],
                    name=act_name
                ))
                prev_output = act_name

        # 3) Définition de la sortie
        output_tensor = helper.make_tensor_value_info(prev_output, TensorProto.FLOAT, None)

        # 4) Construction du graphe et du modèle
        graph = helper.make_graph(
            nodes,
            'InferenceNetwork',
            [input_tensor],
            [output_tensor],
            initializer=initializers
        )
        model = helper.make_model(graph, producer_name='NetworkONNXExporter')
        onnx.save(model, filepath)
