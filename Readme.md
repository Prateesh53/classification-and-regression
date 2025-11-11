# Heart Disease Project — Full Combined Report (Classification, Regression, ONNX)

This report covers data processing, model training, evaluation, and ONNX export for both PyTorch and XGBoost models, all in one unified document.



## Dataset and Preparation
The dataset `heart.csv` contains **918 samples**, **12 columns**, and the target variable **HeartDisease (0/1)**.  
Only **numerical features** were used:

`Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak`

Final shapes:
- Features `X`: (918, 6)
- Labels `y`: (918, 1)

Data was split into training and testing sets (80/20), then converted to `float32` for inputs and `int64` for labels where required. Standardization statistics were computed from training data (mean and std).



## PyTorch Classification Models

### 1. MLP (Single Hidden Layer)
Architecture:
- Linear(6 → 5), ReLU  
- Linear(5 → 2), Softmax  

Training:
- 1000 epochs  
- Adam optimizer  
- LR = 0.0003  

Results:
Accuracy: 0.61
Confusion Matrix:
[[ 0 72]
[ 0 112]]
Precision: 0.371
Recall: 0.609
F1: 0.461



Interpretation:  
The network collapses into predicting a single class. No true negatives occurred. The model underfits heavily.



### 2. Deep Neural Network (Two Hidden Layers)
Architecture:
- Linear(6 → 10), ReLU, Dropout  
- Linear(10 → 5), ReLU, Dropout  
- Linear(5 → 2), Softmax  

Results:
Accuracy: 0.79
Confusion Matrix:
[[60 12]
[27 85]]
Precision: 0.803
Recall: 0.788
F1: 0.790


Interpretation:  
A deeper network successfully models the nonlinear structure and provides strong, balanced performance. This is the **best classification model** in the project.



## PyTorch Regression Models
A regression experiment was performed even though the target is binary.

### Deep Regression Network (6 → 10 → 6 → 1)
Result:


R² = 0.296


Interpretation:  
Regression is mathematically inappropriate for a binary variable. The low R² confirms the mismatch.



## PyTorch → ONNX Export
Export command:
```python
torch.onnx.export(model, dummy_input, "heart.onnx")


Outcome:

Export succeeded.

ONNX model validated correctly.

Warning appeared due to PyTorch’s legacy exporter path.

ONNX Runtime inference works with output shape (1, 1).

Interpretation:
The warning is cosmetic. Export is correct and fully functional.

XGBoost Model and Results

A regression-style XGBoost model was also trained:

XGBRegressor(n_estimators=100, max_depth=3)


Results:

R² = 0.275


Interpretation:
Just like the PyTorch regression model, this is not a suitable framing for a binary label. Predictions are continuous and not clinically meaningful.

XGBoost → ONNX Conversion Errors
Error 1 — “0 feature is supplied”

Cause:
The internal Booster stored no feature metadata.
This happens when the model is created via the raw Booster API or loaded incorrectly.

Fix:
Train via:

regressor.fit(X_train, y_train)

Error 2 — “could not convert string to float: '[5.517711E-1]'”

Cause:
XGBoost 2.x writes base_score as a string inside brackets, but onnxmltools expects a float.

Fix:

pip install "xgboost<2.0"


or switch to conversion via skl2onnx.

Error 3 — sklearn tag errors
AttributeError: '__sklearn_tags__'


Cause:
Jupyter notebook display system fails when printing the estimator.
This does not affect training, prediction, or ONNX conversion.

Successful ONNX Runtime Inference After Fixes

Once XGBoost was downgraded and metadata correctly included, ONNX Runtime produced outputs identical to the original model:

[[0.8793],
 [0.9024],
 ...]


Interpretation:
The ONNX model is consistent with XGBoost predictions after resolving version incompatibility and ensuring valid metadata.

Final Consolidated Conclusions

The Deep Neural Network is the strongest model with ~79% accuracy.

Single-layer MLP suffers severe underfitting.

Regression models (PyTorch and XGBoost) are inappropriate for binary labels.

PyTorch ONNX export works correctly; the warning is harmless.

XGBoost ONNX conversion fails under:

XGBoost ≥ 2.0

Missing feature metadata

onnxmltools parser limitations

Conversion succeeds with:

XGBoost < 2.0

convert_sklearn() or correct estimator training

Proper input shape [None, 6]

# Heart Disease Project — Classification, Regression, ONNX Export, Errors, Fixes (Complete Consolidated Report)

This report contains **all results, explanations, errors, fixes, and conclusions in one continuous markdown document**, without separating into multiple sections.



The dataset used (`heart.csv`) contains 918 rows and 12 columns. Only **numerical columns** were selected for modeling: `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`, giving `X` with shape (918, 6) and `y` with shape (918, 1). The target `HeartDisease` is binary, so classification is the correct task; regression is mathematically invalid for this outcome.

The first model trained was a simple MLP with one hidden layer using PyTorch. It consisted of Linear(6→5), ReLU, Linear(5→2), Softmax. Training for 1000 epochs with LR=0.0003 produced poor results: accuracy 0.61, confusion matrix showing zero true negatives, with precision 0.371, recall 0.609, F1=0.461. The model collapsed into predicting only one class and failed to learn a meaningful boundary.

A deeper neural network was then trained: Linear(6→10), ReLU, Dropout, Linear(10→5), ReLU, Dropout, Linear(5→2), Softmax. This model achieved significantly better performance with accuracy 0.79, confusion matrix [[60,12],[27,85]], precision 0.803, recall 0.788, and F1=0.790. This indicates the deeper network successfully captured nonlinear patterns in the data and was the best-performing classification model.

Regression models were also trained despite the target being binary. A deep regression architecture (6→10→6→1) achieved R² = 0.296, confirming that regression is not suitable for predicting a binary medical label. The continuous outputs were not meaningful, and the model lacked explanatory power for classification.

PyTorch ONNX export was performed using `torch.onnx.export(model, dummy_input, "heart.onnx")`. Export completed successfully, only producing a deprecation warning about the legacy TorchScript exporter. The ONNX model loaded correctly in ONNX Runtime, verified by `onnx.checker.check_model`, and produced output of shape (1,1). Functionally, the export was correct.

An XGBoost model was then trained using `XGBRegressor(n_estimators=100, max_depth=3)`, again in a regression setup. The model produced R² = 0.275 on the test set, consistent with the fact that regression is inappropriate for the binary target. Predictions were continuous values, not probabilities or class labels.

The major difficulties occurred during ONNX export of XGBoost using `onnxmltools`. The first error reported was:

0 feature is supplied. Are you using raw Booster interface?

vbnet
Copy code

This happens when the underlying Booster has no feature metadata. This occurs if a raw Booster was used, if the model was loaded incorrectly, or if metadata was not saved. The fix is to train using the scikit-learn API: `XGBRegressor.fit(X,y)`.

The next error was:

ValueError: could not convert string to float: '[5.517711E-1]'

typescript
Copy code

This originates from XGBoost ≥ 2.0, which stores `base_score` in JSON as a bracketed string, while `onnxmltools` expects a float. Because of this mismatch, conversion fails. The correct fix is:

pip install "xgboost<2.0"

nginx
Copy code

Another error occurred during display:

AttributeError: 'sklearn_tags'

pgsql
Copy code

This error is raised only because Jupyter attempts to pretty-print the model. It does not affect training or ONNX conversion.

Once dependencies were fixed (XGBoost < 2.0, scikit-learn API used, correct initial types), the ONNX model exported successfully. ONNX Runtime inference produced values identical to the original XGBoost predictions, confirming the correctness of the conversion pipeline.

Across the entire assignment, the strongest model was the deep classification network with ~79% accuracy. Regression was consistently inappropriate, and XGBoost → ONNX conversion only works after aligning versions and ensuring the model is trained through the sklearn wrapper. PyTorch ONNX export was correct aside from a migration warning. The major blockers were library incompatibilities, not model logic.



## Final Note

This unified report captures the complete workflow—from dataset preparation to model development, evaluation, ONNX deployment, and debugging of conversion failures—ensuring the full project can be understood, reproduced, and extended without referring to any external segments or fragmented explanations.






