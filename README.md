# PPG_Breath_Detection_Model
A repository for a project pertaining to detecting and localizing inspiration and expiration of breath on a PPG signal input, using Dual Classifier-Regressor architecture.

This project aims to design a Dual-Head Neural Network for the localization of Respiration Phase from Photoplethysmography, and verify such via bench-marking against both Breath per Minute (BPM) scores and Instantaneous Respiratory Rates.
## Model Architecture
### 1. Preprocessing
We use the BIDMC dataset to feed PPG, ECG, and Respiration Impedance signals as inputs, down-sampled to 100Hz and filtered as follows:
- PPG Signal (0.1 to 0.5 Hz)
- ECG Signal (1 to 45 Hz)
The signals are then normalized using 98th percentile, and R-peaks are extracted using generic find_peaks function.
Ground truth labels are extracted from the Respiration Signal by finding peaks and troughs of the signal.
## 2. Dataset Loading
Preprocessing is performed on the datasets, and ground truth labels are encoded in both probability and localization. There is the added functionality of augmented the training set, since the BIDMC consists of ICU patients under controlled environments. The WESAD test set contains noisier signals taken from wearables, so augmentation exists to improve the training process if required.

The dataloader uses a weighted random sampler for building the train set in order to provide a balance between windows with and without a transition in breath state. The test set will remain in default order.
## 3. Model Design
The model consists of the following pipeline:
1. Input Signal Encoder
2. Core Network with Dense Layers
3. Classifier Head
4. Regressor Head

The model contains two pathways - PPG with the auxiliary signals and PPG Only pathway. 
This is done to avoid performance loss in the situation of noisy ECG signal inputs. 

The classification head predicts probabilities for a window to contain a transition of a certain type, and the Regression head localizes this transition in time (if present). The probability is the average of two pathways used.

We use BCEwithLogitsLoss for the Classification Loss and MaskedMSE Loss for the regression loss (we want to avoid regression loss calculation when there is no transition detected so we mask it).

## 4. Model Training [Work in Progress]
The overall loss function is a composite of Classification and Regression loss as follows:
```math
\text{Loss} = \alpha \cdot L_{cls} + (1 - \alpha) \cdot L_{reg}
```
Model training is currently in progress, we are splitting batches and accumulating for easier operation on the limited hardware capabilities of Kaggle / Google CoLab.
## 5. Model Inference [Work in Progress]
We aim to implement an adaptive stitching mechanism for the predicted probabilities to create a coherent timeline of breath states. Performance can be evaluated by using MAE, STD, F1 scores, precision and recall.
## Future Timeline
- [ ] Perform Model parameter tuning
- [ ] Perform Model training on BIDMC training data
- [ ] Perform Model inference on WESAD dataset
- [ ] Visualize Performance metrics (Confusion matrices, plots of PPV, etc.)
- [ ] Extending to bench-marking against other papers and models
## Potential Improvements
There are several additional ideas I wish to implement:
- Initial testing without Augmenting the BIDMC dataset and comparing performance with and without
- Use of better R-Peak detection algorithm, as found in the Paper A Novel R-Peak Detection Algorithm
- Use of Respiration Impedance Derivative for Ground Truth labeling instead of simple peak detection
- Calculation of Breaths per Minute for bench-marking against the Paper RRWaveNet
- Bench-marking against Respiration Phase localization methods in Signal Processing (ideally we must optimize the model to display good performance for both classification (aiding BPM calculation) and localization)
