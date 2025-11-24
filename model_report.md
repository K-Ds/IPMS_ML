# IPMS: Audio Classification Model Report

## 1. Problem Statement
As part of the **Intelligent Pollution Monitoring System (IPMS)**, the objective was to develop a robust machine learning component capable of identifying specific sources of noise pollution in an urban environment.

The key constraint was **edge deployment**: the model must run efficiently on an **ESP32 microcontroller** (TinyML). This required balancing high classification accuracy with a minimal memory footprint and low computational latency.

## 2. Dataset & Preprocessing
The model was trained on the **UrbanSound8K** dataset, a standard benchmark for urban sound classification.

### Data Pipeline
* **Selection:** 8 relevant classes were selected (Car Horn, Dog Bark, Drilling, Engine Idling, Gun Shot, Jackhammer, Siren, Street Music).
* **Audio Formatting:** All clips were resampled to **16kHz** and fixed to a duration of **4 seconds**.
* **Feature Extraction:** Raw audio was converted into **Log Mel-Spectrograms** to capture frequency patterns over time.
    * *Mel Bands:* 64
    * *FFT Size:* 2048
    * *Hop Length:* 512
* **Normalization:** A crucial step for model stability. We calculated the global Mean and Standard Deviation of the training set and applied Z-score normalization.
    * *Note:* These constants (`normalization_constants.json`) are exported for use in the C++ firmware to ensure the ESP32 preprocesses real-world audio exactly like the training data.

## 3. Model Architecture
A custom Convolutional Neural Network (CNN) was designed specifically for this task. Unlike standard pre-trained models (like VGG or ResNet) which are too heavy for an ESP32, this architecture is lightweight.

### Key Architectural Decisions:
1.  **Spatial Dropout:** Instead of standard dropout, `SpatialDropout2D` was used. In spectrograms, adjacent pixels are highly correlated; dropping entire feature maps forces the network to learn more robust features.
2.  **Global Average Pooling (GAP):** Traditional CNNs end with massive Dense layers that consume megabytes of memory. We used GAP to reduce the 3D feature tensor directly to a 1D vector, significantly reducing the parameter count.
3.  **L2 Regularization:** Applied to the final dense layer to constrain weight magnitude, which helps maintain accuracy when the model is later quantized to 8-bit integers.

**Model Stats:**
* **Input Shape:** `(126, 64, 1)`
* **Total Parameters:** ~122,000
* **Estimated Size (Float32):** ~480 KB

## 4. Training Strategy
* **Optimizer:** Adam with an initial learning rate of `1e-3`.
* **Callbacks:**
    * `ModelCheckpoint`: To save only the best performing weights based on validation loss.
    * `ReduceLROnPlateau`: Dynamic learning rate adjustment to fine-tune convergence.
    * `EarlyStopping`: To prevent overfitting if validation loss stagnates.
* **Validation Strategy:** A strict 10-fold cross-validation setup was respected, using Fold 10 specifically for validation to ensure no data leakage.

## 5. Evaluation & Results
The model was evaluated on the unseen data from Fold 10.

### Metrics
* **Overall Accuracy:** **86.8%**
* **Top-3 Accuracy:** **99.3%** (The model nearly always has the correct class in its top 3 guesses).

### Class-wise Performance
As seen in the classification report (saved in `results/classification.png`), the model performs exceptionally well on distinct sounds like **Gun Shots** and **Sirens** (high F1-scores). Continuous noises like **Drilling** proved slightly harder to distinguish from similar mechanical sounds, but performance remains robust.


*(Refer to `results/confusion matrix.png` in the repo for visual breakdown)*

## 6. Challenges & Solutions
Developing for TinyML presents unique difficulties not found in standard server-side AI.

### A. The "Right Size" Dilemma
**Challenge:** Finding a model architecture that was deep enough to learn complex audio features but small enough not to "clash" with the ESP32's limited RAM. Standard architectures often have millions of parameters, which would cause the microcontroller to crash (Stack Overflow/Out of Memory) immediately.
**Solution:** We iterated through several custom architectures. The breakthrough was implementing **Global Average Pooling**. This allowed us to remove the heavy fully-connected layers found in typical CNNs, reducing the model size by over 90% while maintaining the same accuracy.

### B. DSP Consistency (Python vs. C++)
**Challenge:** The model expects input data (Spectrograms) to look *exactly* the same during inference as it did during training. However, the Python library (`librosa`) used for training handles mathematics differently than the C++ DSP libraries used on the ESP32.
**Solution:** We strictly standardized the FFT parameters (Window size 2048, Hop 512) and exported the specific **Normalization Constants** (Mean: -38.72, Std: 22.35) to a JSON file. The C++ firmware reads these values to perform Z-score normalization identically to the Python training pipeline.

### C. Quantization Accuracy Drop
**Challenge:** Converting the model from 32-bit Floating Point numbers to 8-bit Integers (required for speed on ESP32) initially caused a drop in accuracy because the model weights lost precision.
**Solution:** We introduced **L2 Regularization** during the training phase. This penalizes large weights, forcing the model to keep its internal values within a smaller range, making the transition to 8-bit integers smoother and preserving accuracy.

## 7. TinyML Optimization (Quantization)
To run on the ESP32, the model underwent **Post-Training Quantization**.
* **Method:** Full Integer Quantization (INT8).
* **Representative Dataset:** 1000 samples from the training set were used to calibrate the dynamic range of the activations.
* **Outcome:** The final `.tflite` model is approximately **4x smaller** than the Keras version, with minimal loss in accuracy, making it suitable for the limited RAM of the ESP32-S3.

## 8. Conclusion & Future Work
The audio classification module successfully meets the requirements of the IPMS project. It provides accurate, real-time noise categorization on edge hardware.

**Future Improvements:**
* **Data Augmentation:** Adding background city noise to training samples to improve robustness in noisy environments.
* **Continuous Learning:** Implementing a mechanism to flag low-confidence predictions on the ESP32 for manual review and retraining.
