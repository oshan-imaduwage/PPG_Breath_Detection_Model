# PPG_Breath_Detection_Model

This project attempts to automate the segmentation of respiratory phase transitions (inspiration and expiration onsets) using photoplethysmogram (PPG) and electrocardiogram (ECG) signals. I adapted a joint classifier-regressor architecture originally designed for heart sound detection to process low-frequency respiratory data. While the classification pathway successfully identified the presence of breaths (achieving an F1 score of ~0.93), the regression pathway encountered a fundamental mathematical bottleneck, resulting in Mean Collapse.

## Inspirations
This architecture was heavily inspired by the paper ["A Dual Classifier-Regressor Architecture for Heart Sound Onset/Offset Detection"](https://ieeexplore.ieee.org/abstract/document/11355434).
*   **Original Design:** The source architecture processed 400 ms windows of PCG and ECG data to identify S1 and S2 heart sounds. It utilized separate R-peak and T-peak encoders and relied on a Flatten + Dense network block to output exact sample locations.
*   **Our Adaptation:** I ported this architecture to detect respiratory phases and swapped the PCG input for PPG and increased the window size to 3.0 seconds to capture slower respiratory cycles. Because respiration lacks a strict lock to the T-wave, we utilized only a single auxiliary R-peak marker alongside the primary signals.

## Model Architecture 

<img src="https://github.com/oshan-imaduwage/PPG_Breath_Detection_Model/blob/4d3e7179e5251f1fdd3dbb51a4cbca57fd6357a0/Images/ppg_model.onnx.png" width="500"/>

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

I use BCEwithLogitsLoss for the Classification Loss and MaskedMSE Loss for the regression loss (I want to avoid regression loss calculation when there is no transition detected so it is masked).

## 4. Model Training
The overall loss function is a composite of Classification and Regression loss as follows:

```math
\text{Loss} = \alpha \cdot L_{cls} + (1 - \alpha) \cdot L_{reg}
```


## Training and Hyperparameter Tuning
I performed hyperparameter tuning by both manually tweaking parameters of significance (i.e. Window Sizes, Alpha, etc.) and by utilizing Optuna for multi-objective hyperparameter tuning (maximizing F1 while minimizing MAE) across Kaggle and Google Colab environments. The training notebooks are available in the `Model_Notebooks/` directory. 

Despite implementing gradient scaling, Huber loss (Smooth L1), and LayerNorm fixes, the model's regressor failed to converge properly, as seen below with different window sizes:

   1                       |  2
:-------------------------:|:-------------------------:
![](https://github.com/oshan-imaduwage/PPG_Breath_Detection_Model/blob/4d3e7179e5251f1fdd3dbb51a4cbca57fd6357a0/Images/trainwin2.png) |  ![](https://github.com/oshan-imaduwage/PPG_Breath_Detection_Model/blob/4d3e7179e5251f1fdd3dbb51a4cbca57fd6357a0/Images/trainwin3.png)

## Key Discoveries:
Diagnostic testing revealed that the regressor suffered from "Mean Collapse." On a 3.0-second window (300 samples), the expected MAE of guessing the exact center of the window every time is 750 ms. The diagnostic runs consistently stalled at an MAE of ~714 ms to 749 ms. The model was effectively blind to time, guessing the middle of the window to minimize error.

I conclude that the original architecture is structurally unsuitable for this specific dataset and task due to three factors:

1.  **Physiological Anchor Mismatch:** The original paper relied on the strict physiological coupling where S1 occurs $20 \pm 5$ ms after the R-peak and S2 occurs $24 \pm 6$ ms after the T-wave. Respiration is a low-frequency process that spans multiple cardiac cycles, offering no such deterministic, millisecond-exact anchor to the ECG.
2.  **Spatial Amnesia:** The original network flattened a 400 ms window containing a single R-peak. The 3.0-second window I used contains multiple R-peaks and a slow-rolling PPG wave. Passing this through `MaxPool1d` and `Flatten` operations scrambles the spatial phase data, which I believe makes it impossible for the dense layers to map the features back to a continuous temporal coordinate.
3.  **Missing Features:** The original architecture utilized distinct R-peak and T-peak encoders to differentiate S1 from S2. The respiratory dataset adaptation used lacks a secondary temporal anchor, which may be limiting the regressor's ability to localize specific transition types.

## Future Work
Based on these findings, future iterations of this pipeline should abandon `Flatten` + `Dense` coordinate regression for long-window physiological signals. Viable alternatives may include:
*   **1D Soft-Argmax:** Utilizing a fully convolutional core to generate a temporal heatmap and calculating the expected value (center of mass).
*   **Semantic Segmentation:** Treating the problem as a 1D U-Net segmentation task rather than bounding-box style regression.
