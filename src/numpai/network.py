from __future__ import annotations

import copy
import pickle
import warnings
from collections.abc import Callable, Sequence
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .layers import (
    ActivationLayer,
    BatchNormalization,
    Conv2D,
    Dropout,
    Embedding,
    FCLayer,
    FlattenLayer,
    GlobalAvgPool2D,
    LSTM,
    Layer,
    MaxPooling,
)

LossFunction = Callable[[np.ndarray, np.ndarray], float | np.float64]
LossPrimeFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Network:
    def __init__(self) -> None:
        self.layers: list[Layer] = []
        self.loss: LossFunction | None = None
        self.loss_prime: LossPrimeFunction | None = None
        self.logs: list[float] = []
        self.logs_test: list[float] = []
        self.training_data: dict[str, list[np.ndarray]] = {"X": [], "y": []}
        self._inference_only = False

    def add(self, layer: Layer) -> None:
        self._ensure_training_enabled()
        if not isinstance(layer, Layer):
            raise TypeError(f"layer doit heriter de Layer, recu {type(layer).__name__}")
        self.layers.append(layer)

    def use(self, loss: LossFunction, loss_prime: LossPrimeFunction) -> None:
        self._ensure_training_enabled()
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        return [self._forward(sample, training=False) for sample in input_data]

    def fit(
        self,
        x_train: Sequence[np.ndarray] | np.ndarray,
        y_train: Sequence[np.ndarray] | np.ndarray,
        epochs: int,
        learning_rate: float,
    ) -> None:
        self._warn_deprecated("fit", "fit3")
        self._ensure_training_ready(epochs=epochs, learning_rate=learning_rate)

        samples = len(x_train)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample_index in range(samples):
                output = self._forward(x_train[sample_index], training=True)
                epoch_loss += float(self.loss(y_train[sample_index], output))
                error = self.loss_prime(y_train[sample_index], output)
                self._backward(error, learning_rate, epoch)

            epoch_loss /= samples
            print(f"epoch {epoch + 1}/{epochs} error={epoch_loss:.6f}")

    def fit2(
        self,
        x_train: Sequence[np.ndarray] | np.ndarray,
        y_train: Sequence[np.ndarray] | np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
    ) -> None:
        self._warn_deprecated("fit2", "fit3")
        self._ensure_training_ready(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        for epoch in range(epochs):
            epoch_loss = self._train_epoch(x_train, y_train, learning_rate, batch_size, epoch)
            self.logs.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} error={epoch_loss:.6f}")

    def fit3(
        self,
        x_train: Sequence[np.ndarray] | np.ndarray,
        y_train: Sequence[np.ndarray] | np.ndarray,
        x_val: Sequence[np.ndarray] | np.ndarray,
        y_val: Sequence[np.ndarray] | np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        patience_lr_decay: int = 2,
        patience_early_stopping: int = 5,
        min_lr: float = 1e-5,
    ) -> None:
        self._ensure_training_ready(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        if patience_lr_decay <= 0:
            raise ValueError("patience_lr_decay doit etre strictement positif")
        if patience_early_stopping <= 0:
            raise ValueError("patience_early_stopping doit etre strictement positif")
        if min_lr <= 0:
            raise ValueError("min_lr doit etre strictement positif")

        best_val_loss = float("inf")
        best_layers = copy.deepcopy(self.layers)
        epochs_without_improvement = 0
        epochs_without_lr_decay = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(x_train, y_train, learning_rate, batch_size, epoch)
            val_loss = self._validation_loss(x_val, y_val)
            self.logs.append(train_loss)
            self.logs_test.append(val_loss)

            print(f"epoch {epoch + 1}/{epochs} train_error={train_loss:.6f} val_error={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_layers = copy.deepcopy(self.layers)
                epochs_without_improvement = 0
                epochs_without_lr_decay = 0
                continue

            epochs_without_improvement += 1
            epochs_without_lr_decay += 1

            if epochs_without_lr_decay >= patience_lr_decay and learning_rate > min_lr:
                learning_rate = max(learning_rate * 0.5, min_lr)
                print(f"Nouveau learning rate : {learning_rate:.1e}")
                epochs_without_lr_decay = 0

            if epochs_without_improvement >= patience_early_stopping:
                print(
                    f"Arret precoce apres {epoch + 1} epoques "
                    f"(patience={patience_early_stopping})."
                )
                break

        self.layers = best_layers

    def summary(self) -> None:
        print("\nModel Summary:\n")
        total_params = 0
        for index, layer in enumerate(self.layers, start=1):
            output_shape = self._layer_output_shape(layer)
            params_count = self._layer_parameter_count(layer)
            total_params += params_count
            print(
                f"Layer {index}: {type(layer).__name__} | "
                f"Output Shape: {output_shape} | Parameters: {params_count}"
            )

        print(f"Total Parameters: {total_params}")
        logs = getattr(self, "logs", [])
        logs_test = getattr(self, "logs_test", [])
        if logs:
            plt.plot(logs, marker="o", linestyle="-", color="b", label="Erreur")
            if logs_test:
                plt.plot(logs_test, marker="o", linestyle="-", color="r", label="Erreur de validation")
            plt.title("Erreur au cours du temps (epochs)", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Erreur (UA)", fontsize=12)
            plt.grid(True)
            plt.legend(loc="best")
            plt.show()

    def save(self, filename: str | Path, debug: bool = True) -> None:
        path = Path(filename)
        with path.open("wb") as file:
            pickle.dump(self, file)
        if debug:
            print(f"Model saved to {path}")

    @staticmethod
    def load(filename: str | Path, debug: bool = True) -> Network:
        path = Path(filename)
        with path.open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, Network):
            raise TypeError(f"Le fichier {path} ne contient pas un Network")
        if debug:
            print(f"Model loaded from {path}")
        return model

    def export(self) -> None:
        fused_layers: list[Layer] = []
        for layer in self.layers:
            if isinstance(layer, Dropout):
                continue

            if isinstance(layer, BatchNormalization) and fused_layers:
                previous_layer = fused_layers[-1]
                if isinstance(previous_layer, FCLayer):
                    self._fuse_batchnorm_into_fc(previous_layer, layer)
                    continue
                if isinstance(previous_layer, Conv2D):
                    self._fuse_batchnorm_into_conv(previous_layer, layer)
                    continue

            if isinstance(layer, FCLayer):
                self._strip_fc_optimizer_state(layer)

            fused_layers.append(layer)

        self.layers = fused_layers
        for attr in ("loss", "loss_prime", "logs", "logs_test", "training_data"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._inference_only = True

    def _forward(self, input_data: np.ndarray, training: bool) -> np.ndarray:
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output, training=training)
        return output

    def _backward(self, error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray | None:
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate, epoch)
        return error

    def _train_epoch(
        self,
        x_train: Sequence[np.ndarray] | np.ndarray,
        y_train: Sequence[np.ndarray] | np.ndarray,
        learning_rate: float,
        batch_size: int,
        epoch: int,
    ) -> float:
        samples = len(x_train)
        epoch_loss = 0.0
        batches = ceil(samples / batch_size)

        for batch_index, batch_start in enumerate(range(0, samples, batch_size), start=1):
            batch_end = batch_start + batch_size
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            output = self._forward(x_batch, training=True)
            epoch_loss += float(self.loss(y_batch, output))
            batch_error = self.loss_prime(y_batch, output)
            self._backward(batch_error, learning_rate, epoch)

            print(f"\rtraining batch {batch_index}/{batches}", end="")

        print("\r", end="")
        return epoch_loss / samples

    def _validation_loss(
        self,
        x_val: Sequence[np.ndarray] | np.ndarray,
        y_val: Sequence[np.ndarray] | np.ndarray,
    ) -> float:
        val_output = np.asarray(self.predict(x_val))
        return float(self.loss(np.asarray(y_val), val_output)) / len(x_val)

    def _ensure_training_ready(
        self,
        epochs: int,
        learning_rate: float,
        batch_size: int | None = None,
    ) -> None:
        self._ensure_training_enabled()
        self._ensure_loss_configured()
        if epochs <= 0:
            raise ValueError("epochs doit etre strictement positif")
        if learning_rate <= 0:
            raise ValueError("learning_rate doit etre strictement positif")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size doit etre strictement positif")

    def _ensure_training_enabled(self) -> None:
        if self._inference_only:
            raise RuntimeError("Ce Network a ete exporte pour l'inference et ne peut plus etre entraine.")

    def _ensure_loss_configured(self) -> None:
        if self.loss is None or self.loss_prime is None:
            raise ValueError("Appelez use(loss, loss_prime) avant l'entrainement.")

    def _warn_deprecated(self, method_name: str, replacement: str) -> None:
        warnings.warn(
            f"'{method_name}' est deprecie et sera supprime. Utilisez '{replacement}' a la place.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _layer_output_shape(self, layer: Layer) -> tuple[int, ...] | str | int:
        if isinstance(layer, Conv2D):
            input_height, input_width = layer.input_shape[1], layer.input_shape[2]
            filter_height, filter_width = layer.filters.shape[2], layer.filters.shape[3]
            output_height = (input_height - filter_height + 2 * layer.padding) // layer.stride + 1
            output_width = (input_width - filter_width + 2 * layer.padding) // layer.stride + 1
            return (layer.num_filters, output_height, output_width)
        if isinstance(layer, FCLayer):
            return (layer.biases.shape[1],)
        if isinstance(layer, BatchNormalization):
            return layer.input_size
        return getattr(layer, "output_size", "N/A")

    def _layer_parameter_count(self, layer: Layer) -> int:
        if isinstance(layer, Conv2D):
            return int(layer.filters.size + layer.biases.size)
        if isinstance(layer, FCLayer):
            return int(layer.weights.size + layer.biases.size)
        if isinstance(layer, BatchNormalization):
            return int(layer.gamma.size + layer.beta.size)

        params_count = 0
        for attr in ("weights", "biases", "embeddings", "W", "b", "gamma", "beta"):
            value = getattr(layer, attr, None)
            if isinstance(value, np.ndarray):
                params_count += int(value.size)
        return params_count

    def _fuse_batchnorm_into_fc(self, layer: FCLayer, batchnorm: BatchNormalization) -> None:
        mean = batchnorm.running_mean.flatten()
        variance = batchnorm.running_var.flatten()
        gamma = batchnorm.gamma.flatten()
        beta = batchnorm.beta.flatten()
        scale = gamma / np.sqrt(variance + batchnorm.epsilon)

        layer.weights *= scale[np.newaxis, :]
        layer.biases = (layer.biases - mean) * scale + beta

    def _fuse_batchnorm_into_conv(self, layer: Conv2D, batchnorm: BatchNormalization) -> None:
        filters_count = layer.filters.shape[0]
        mean = batchnorm.running_mean.reshape(filters_count, 1, 1)
        variance = batchnorm.running_var.reshape(filters_count, 1, 1)
        gamma = batchnorm.gamma.reshape(filters_count, 1, 1)
        beta = batchnorm.beta.reshape(filters_count, 1, 1)
        scale = gamma / np.sqrt(variance + batchnorm.epsilon)

        layer.filters *= scale[:, np.newaxis, :, :]
        layer.biases = (layer.biases - mean) * scale + beta

    def _strip_fc_optimizer_state(self, layer: FCLayer) -> None:
        for attr in ("m_weights", "v_weights", "m_biases", "v_biases", "beta1", "beta2", "epsilon", "t"):
            if hasattr(layer, attr):
                delattr(layer, attr)

    def export_to_onnx(self, filepath: str | Path, input_shape: tuple[int | None, ...]) -> None:
        """
        Export the inference-optimized network to ONNX format.

        Parameters
        ----------
        filepath : str
            Path to save the .onnx model file. Utilise par `onnx.save`.
        input_shape : tuple of int
            Shape of the input tensor, e.g., (batch_size, channels, height, width).
            Passe a `helper.make_tensor_value_info` pour definir la signature.

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
        from onnx import TensorProto, helper

        nodes = []
        initializers = []
        input_name = "input"
        previous_output = input_name
        current_shape = list(input_shape)

        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, self._onnx_shape(input_shape))

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                output_name = f"conv_{index}_output"
                weight_name = f"conv_{index}_weights"
                bias_name = f"conv_{index}_biases"
                initializers.append(self._onnx_tensor(helper, TensorProto, weight_name, layer.filters))
                initializers.append(self._onnx_tensor(helper, TensorProto, bias_name, layer.biases.reshape(-1)))
                nodes.append(
                    helper.make_node(
                        "Conv",
                        inputs=[previous_output, weight_name, bias_name],
                        outputs=[output_name],
                        name=f"Conv_{index}",
                        pads=[layer.padding, layer.padding, layer.padding, layer.padding],
                        strides=[layer.stride, layer.stride],
                    )
                )
                previous_output = output_name
                current_shape = [
                    current_shape[0],
                    layer.num_filters,
                    self._onnx_conv_output_dim(current_shape[2], layer.filter_size, layer.padding, layer.stride),
                    self._onnx_conv_output_dim(current_shape[3], layer.filter_size, layer.padding, layer.stride),
                ]
                continue

            if isinstance(layer, FCLayer):
                if len(current_shape) != 2:
                    raise ValueError("FCLayer attend une entree ONNX 2D. Ajoutez FlattenLayer avant FCLayer.")
                output_name = f"fc_{index}_output"
                weight_name = f"fc_{index}_weights"
                bias_name = f"fc_{index}_biases"
                initializers.append(self._onnx_tensor(helper, TensorProto, weight_name, layer.weights))
                initializers.append(self._onnx_tensor(helper, TensorProto, bias_name, layer.biases.reshape(-1)))
                nodes.append(
                    helper.make_node(
                        "Gemm",
                        inputs=[previous_output, weight_name, bias_name],
                        outputs=[output_name],
                        name=f"Gemm_{index}",
                    )
                )
                previous_output = output_name
                current_shape = [current_shape[0], layer.biases.shape[1]]
                continue

            if isinstance(layer, ActivationLayer):
                previous_output = self._append_onnx_activation_node(helper, nodes, layer, previous_output, index)
                continue

            if isinstance(layer, FlattenLayer):
                if len(current_shape) < 2:
                    raise ValueError("FlattenLayer attend une entree ONNX avec une dimension batch.")
                output_name = f"flatten_{index}_output"
                nodes.append(
                    helper.make_node(
                        "Flatten",
                        inputs=[previous_output],
                        outputs=[output_name],
                        name=f"Flatten_{index}",
                        axis=1,
                    )
                )
                previous_output = output_name
                current_shape = [current_shape[0], self._onnx_product(current_shape[1:])]
                continue

            if isinstance(layer, MaxPooling):
                if len(current_shape) != 4:
                    raise ValueError("MaxPooling attend une entree ONNX 4D.")
                output_name = f"maxpool_{index}_output"
                nodes.append(
                    helper.make_node(
                        "MaxPool",
                        inputs=[previous_output],
                        outputs=[output_name],
                        name=f"MaxPool_{index}",
                        kernel_shape=[layer.pool_size, layer.pool_size],
                        strides=[layer.stride, layer.stride],
                    )
                )
                previous_output = output_name
                current_shape = [
                    current_shape[0],
                    current_shape[1],
                    self._onnx_conv_output_dim(current_shape[2], layer.pool_size, 0, layer.stride),
                    self._onnx_conv_output_dim(current_shape[3], layer.pool_size, 0, layer.stride),
                ]
                continue

            if isinstance(layer, GlobalAvgPool2D):
                if len(current_shape) != 4:
                    raise ValueError("GlobalAvgPool2D attend une entree ONNX 4D.")
                pooled_output = f"global_avg_pool_{index}_output"
                flattened_output = f"global_avg_pool_{index}_flattened"
                nodes.append(
                    helper.make_node(
                        "GlobalAveragePool",
                        inputs=[previous_output],
                        outputs=[pooled_output],
                        name=f"GlobalAveragePool_{index}",
                    )
                )
                nodes.append(
                    helper.make_node(
                        "Flatten",
                        inputs=[pooled_output],
                        outputs=[flattened_output],
                        name=f"GlobalAveragePoolFlatten_{index}",
                        axis=1,
                    )
                )
                previous_output = flattened_output
                current_shape = [current_shape[0], current_shape[1]]
                continue

            if isinstance(layer, BatchNormalization):
                output_name = f"batchnorm_{index}_output"
                scale_name = f"batchnorm_{index}_scale"
                bias_name = f"batchnorm_{index}_bias"
                mean_name = f"batchnorm_{index}_mean"
                variance_name = f"batchnorm_{index}_variance"
                initializers.extend(
                    [
                        self._onnx_tensor(helper, TensorProto, scale_name, layer.gamma.reshape(-1)),
                        self._onnx_tensor(helper, TensorProto, bias_name, layer.beta.reshape(-1)),
                        self._onnx_tensor(helper, TensorProto, mean_name, layer.running_mean.reshape(-1)),
                        self._onnx_tensor(helper, TensorProto, variance_name, layer.running_var.reshape(-1)),
                    ]
                )
                nodes.append(
                    helper.make_node(
                        "BatchNormalization",
                        inputs=[previous_output, scale_name, bias_name, mean_name, variance_name],
                        outputs=[output_name],
                        name=f"BatchNormalization_{index}",
                        epsilon=layer.epsilon,
                        momentum=layer.momentum,
                    )
                )
                previous_output = output_name
                continue

            if isinstance(layer, Dropout):
                continue

            if isinstance(layer, Embedding | LSTM):
                raise NotImplementedError(f"Export ONNX non supporte pour {type(layer).__name__}")

            raise NotImplementedError(f"Export ONNX non supporte pour {type(layer).__name__}")

        output_tensor = helper.make_tensor_value_info(previous_output, TensorProto.FLOAT, self._onnx_shape(current_shape))
        graph = helper.make_graph(
            nodes,
            "NumpaiNetwork",
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            producer_name="numpai",
            opset_imports=[helper.make_operatorsetid("", 13)],
        )
        onnx.checker.check_model(model)
        onnx.save(model, filepath)

    def _append_onnx_activation_node(self, helper, nodes: list, layer: ActivationLayer, previous_output: str, index: int) -> str:
        activation_name = layer.activation.__name__.lower()
        output_name = f"activation_{index}_output"

        if activation_name == "relu":
            nodes.append(helper.make_node("Relu", inputs=[previous_output], outputs=[output_name], name=f"Relu_{index}"))
            return output_name
        if activation_name == "tanh":
            nodes.append(helper.make_node("Tanh", inputs=[previous_output], outputs=[output_name], name=f"Tanh_{index}"))
            return output_name
        if activation_name == "sigmoid":
            nodes.append(helper.make_node("Sigmoid", inputs=[previous_output], outputs=[output_name], name=f"Sigmoid_{index}"))
            return output_name
        if activation_name == "leakyrelu":
            nodes.append(
                helper.make_node(
                    "LeakyRelu",
                    inputs=[previous_output],
                    outputs=[output_name],
                    name=f"LeakyRelu_{index}",
                    alpha=0.01,
                )
            )
            return output_name
        if activation_name == "softmax":
            nodes.append(
                helper.make_node(
                    "Softmax",
                    inputs=[previous_output],
                    outputs=[output_name],
                    name=f"Softmax_{index}",
                    axis=-1,
                )
            )
            return output_name
        if activation_name == "swish":
            sigmoid_output = f"activation_{index}_sigmoid"
            nodes.append(
                helper.make_node(
                    "Sigmoid",
                    inputs=[previous_output],
                    outputs=[sigmoid_output],
                    name=f"SwishSigmoid_{index}",
                )
            )
            nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[previous_output, sigmoid_output],
                    outputs=[output_name],
                    name=f"SwishMul_{index}",
                )
            )
            return output_name

        raise NotImplementedError(f"Activation non supportee pour ONNX: {activation_name}")

    def _onnx_tensor(self, helper, TensorProto, name: str, value: np.ndarray):
        tensor = np.asarray(value, dtype=np.float32)
        return helper.make_tensor(name, TensorProto.FLOAT, list(tensor.shape), tensor.flatten().tolist())

    def _onnx_shape(self, shape: Sequence[int | None]) -> list[int | str]:
        return [dim if dim is not None else f"dim_{index}" for index, dim in enumerate(shape)]

    def _onnx_product(self, dims: Sequence[int | None]) -> int | None:
        product = 1
        for dim in dims:
            if dim is None:
                return None
            product *= dim
        return product

    def _onnx_conv_output_dim(self, dim: int | None, kernel_size: int, padding: int, stride: int) -> int | None:
        if dim is None:
            return None
        return (dim - kernel_size + 2 * padding) // stride + 1
