"""DNN classifier utilities for the lead-generation pipeline."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

log = logging.getLogger(__name__)


def _require_dnn_dependencies():
    try:
        import tensorflow as tf
        from scikeras.wrappers import KerasClassifier
    except Exception as exc:
        raise RuntimeError(
            "DNN backend requires tensorflow and scikeras. "
            "Install requirements-dnn.txt."
        ) from exc
    return tf, KerasClassifier


def _normalize_hidden_units(hidden_units: Iterable[int] | int) -> list[int]:
    if isinstance(hidden_units, int):
        return [hidden_units]
    if hidden_units is None:
        return []
    return list(hidden_units)


def build_model(
    meta,
    hidden_units: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    l2: float = 0.0,
    learning_rate: float = 1e-3,
):
    tf, _ = _require_dnn_dependencies()

    n_features = meta.get("n_features_in_")
    if not n_features:
        raise ValueError("Missing n_features_in_ in SciKeras meta.")

    random_state = meta.get("random_state")
    if random_state is not None:
        tf.keras.utils.set_random_seed(random_state)

    reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    inputs = tf.keras.Input(shape=(n_features,), dtype=tf.float32)
    x = inputs
    for units in _normalize_hidden_units(hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def make_dnn_estimator(
    hidden_units: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    l2: float = 0.0,
    learning_rate: float = 1e-3,
    batch_size: int = 4096,
    epochs: int = 20,
    patience: int = 3,
    validation_split: float = 0.1,
    random_state: int = 42,
    verbose: int = 0,
):
    tf, KerasClassifier = _require_dnn_dependencies()

    callbacks = []
    if patience and patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    return KerasClassifier(
        model=build_model,
        model__hidden_units=hidden_units,
        model__dropout=dropout,
        model__l2=l2,
        model__learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        random_state=random_state,
        verbose=verbose,
        fit__validation_split=validation_split,
        fit__shuffle=False,
        fit__callbacks=callbacks,
    )
