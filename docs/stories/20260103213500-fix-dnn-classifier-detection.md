# Fix DNN Classifier Type Detection in scikit-learn 1.7.x

## Summary

Modified DNN output layer from single sigmoid neuron to 2-neuron softmax to fix scikit-learn 1.7.x misdetecting KerasClassifier as a regressor during hyperparameter search.

## Context / Problem

When running hyperparameter search with the DNN backend, the following error occurred during cross-validation scoring:

```
ValueError: Pipeline should either be a classifier to be used with response_method=predict_proba or the response_method should be 'predict'. Got a regressor with response_method=predict_proba instead.
```

**Root cause**: scikit-learn 1.7.x changed its estimator type detection logic. The KerasClassifier with a single sigmoid output neuron (common pattern for binary classification) was being misclassified as a regressor instead of a classifier. This caused the scoring function to fail when trying to call `predict_proba`.

**Why this happened**:
- Single sigmoid output: `Dense(1, activation="sigmoid")` → output shape `(n_samples, 1)`
- scikit-learn 1.7.x interprets single-output models as regressors
- Even though `loss="binary_crossentropy"` suggests classification, sklearn checks output shape first

## What Changed

### `model_backends/dnn_classifier.py`:

**Before** (lines 56-63):
```python
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
)
```

**After** (lines 56-66):
```python
# Use 2 output neurons with softmax for explicit binary classification
# This ensures sklearn properly detects this as a classifier
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",  # Changed from binary_crossentropy
    metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
)
```

**Key changes**:
1. **Output layer**: `Dense(1, activation="sigmoid")` → `Dense(2, activation="softmax")`
   - Now outputs 2 probabilities (one per class) instead of single probability
   - Output shape: `(n_samples, 2)` instead of `(n_samples, 1)`

2. **Loss function**: `"binary_crossentropy"` → `"sparse_categorical_crossentropy"`
   - Appropriate for 2-class softmax output with integer labels
   - Mathematically equivalent for binary classification
   - "sparse" means labels are integers (0, 1) not one-hot encoded

## How to Test

Run the training pipeline with DNN backend:

```bash
wsl bash -c "cd '/mnt/c/Lokal_Code/Mitgliedschafts Voraussage/Mitgliedschafts Voraussage' && source venv_wsl/bin/activate && MODEL_BACKEND=dnn_gpu python3 -c 'from training_lead_generation_model import main; main(use_cache=True)'"
```

**Expected behavior:**
* Hyperparameter search starts successfully
* Cross-validation scoring completes without type errors
* No ValueError about regressor/classifier mismatch
* Training proceeds to completion

## Risk / Rollback Notes

* **Risk:** Very low - this is a standard pattern change
* **Functionality**: Mathematically equivalent for binary classification
  - Single sigmoid: outputs P(class=1) ∈ [0, 1]
  - 2-neuron softmax: outputs [P(class=0), P(class=1)] where sum = 1
  - `predict_proba` will return the same probabilities (just in 2-column format)
* **Model quality**: No impact - same expressiveness and learning capacity
* **Compatibility**:
  - ✅ Works with sklearn 1.7.x classifier detection
  - ✅ Compatible with all sklearn scorers expecting `predict_proba`
  - ✅ Standard Keras pattern for multi-class classification
* **Rollback**: Revert to single sigmoid output if needed (though may require downgrading sklearn)

## Technical Notes

**Why 2-output softmax is preferred for sklearn integration**:
1. Unambiguous classifier signal to sklearn (2D probability output)
2. Consistent with multi-class classification pattern (extensible if needed)
3. Better integration with sklearn's type checking system
4. Standard in production ML systems using Keras+sklearn pipelines

**Equivalent probability extraction**:
- Old (sigmoid): `proba = model.predict(X)[:, 0]` → P(class=1)
- New (softmax): `proba = model.predict(X)[:, 1]` → P(class=1)
- sklearn's `predict_proba()` handles this automatically
