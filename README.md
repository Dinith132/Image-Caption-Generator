<xaiArtifact artifact_id="8e769b26-3e9d-4f98-8325-d44ab0459081" artifact_version_id="9f691653-3d1f-4575-94f1-0b3c1f87e0af" title="README.md" contentType="text/markdown">

# Image Caption Generator

This project focuses on developing an **Image Caption Generator** using a deep learning approach, combining both a custom-built Convolutional Neural Network (CNN) and the pre-trained **VGG16** model for image feature extraction, paired with a Long Short-Term Memory (LSTM) network for caption generation. The system takes an image as input and generates a meaningful textual description, bridging computer vision and natural language processing.

## Project Overview
The Image Caption Generator combines computer vision and natural language processing to generate human-like captions for images. The model leverages a hybrid approach:
- A **custom CNN** and the **VGG16** model are used to extract visual features from images.
- An **LSTM** network generates coherent captions based on these features.

The project uses a dataset of **8,090 images**, each paired with five human-written captions, to train and evaluate the model. The performance is optimized through hyperparameter tuning and evaluated using BLEU scores.

## Dataset
- **Size**: 8,090 real-world images in `.jpg` format.
- **Annotations**: Each image has five captions describing people, animals, and everyday activities.
- **Storage**: Captions are stored in `captions.txt`.
- **Preprocessing**: Images are resized to 224×224 pixels; captions are cleaned, tokenized, and padded.

## Model Architecture
The model integrates both a custom-built CNN and the pre-trained VGG16 model with an LSTM for caption generation:
- **Custom CNN**: A tailored CNN with convolutional, pooling, and dense layers to extract a 256-dimensional feature vector from images.
- **VGG16**: A pre-trained CNN used to extract a 4096-dimensional feature vector, leveraging its robust feature extraction capabilities.
- **LSTM**: Processes tokenized captions through an embedding layer and generates sequences word by word.
- **Fusion**: Image features (from both CNN and VGG16) and text embeddings are merged and passed through a dense layer with a softmax output to predict the next word.
- **Loss Function**: Categorical cross-entropy.
- **Optimizer**: Adam.

## Data Preprocessing
- **Images**:
  - Resized to 224×224 pixels and normalized.
  - Features extracted using both custom CNN (256-dimensional) and VGG16 (4096-dimensional).
- **Captions**:
  - Cleaned (lowercase, remove non-alphabetic characters).
  - Added `<start>` and `<end>` tokens.
  - Tokenized and converted to integer sequences.
  - Padded to uniform length.
  - One-hot encoded for training.

## Methodology
1. Resize and preprocess images to 224×224 pixels.
2. Extract features using both a custom CNN and VGG16.
3. Clean and tokenize captions, adding `<start>` and `<end>` tokens.
4. Build vocabulary and encode captions into integer sequences.
5. Construct the CNN-LSTM model:
   - Dense layer for image features (custom CNN and VGG16).
   - Embedding + LSTM for text sequences.
   - Merge visual and textual paths.
6. Train on (image + partial caption → next word) pairs.
7. Evaluate using BLEU scores and visual comparisons.

## Model Training and Predictions
- **Training**:
  - Images are processed through both custom CNN and VGG16 to generate feature vectors.
  - Captions are tokenized, indexed, and padded.
  - Model learns to predict the next word given an image and partial caption.
  - Trained with categorical cross-entropy loss and Adam optimizer over multiple epochs.
- **Predictions**:
  - Input image is processed to obtain feature vectors.
  - Caption generation starts with `<start>` token and proceeds word by word until `<end>` or max length.
  - BLEU-1 and BLEU-2 scores evaluate caption quality.

## Hyperparameter Tuning
Hyperparameters tuned using **Keras Tuner** to optimize validation loss:
- **Embedding Size**: 256 or 512.
- **LSTM Units**: Number of neurons in the LSTM layer.
- **Dropout Rate**: To prevent overfitting.
- **Batch Size**: Affects training stability and speed.
- **Learning Rate**: Controls weight update magnitude.
- **Epochs**: Number of passes through the dataset.

## Cross Validation
- Dataset split: 80% training, 20% validation.
- Validation set monitors performance and guides hyperparameter tuning.
- Fixed split used instead of k-fold due to resource constraints of CNN-LSTM training.

## Model Evaluation
- **Quantitative**: BLEU scores measure caption accuracy and fluency.
- **Qualitative**: Visual comparison of generated captions with human-written ones on unseen test images.

## Dependencies
- Python 3.8+
- TensorFlow/Keras
- NumPy
- Pandas
- NLTK (for BLEU score calculation)
- Pillow (for image processing)
- Keras Tuner

## How to Run
1. Clone the repository:
   ```bash:disable-run
   git clone https://github.com/your-username/image-caption-generator.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Place the 8,090 images and `captions.txt` in the `data/` directory.
4. Run preprocessing and training:
   ```bash
   python train.py
   ```
5. Generate captions for new images:
   ```bash
   python predict.py --image path/to/image.jpg
   ```

## Results
- The model generates coherent and contextually relevant captions.
- BLEU scores indicate strong alignment with human-written captions.
- Combining custom CNN and VGG16 improves feature extraction robustness compared to using either alone.

## Future Improvements
- Experiment with additional pre-trained models (e.g., ResNet, Inception).
- Incorporate attention mechanisms to focus on specific image regions.
- Expand the dataset for better generalization.
- Explore transformer-based architectures for caption generation.

</xaiArtifact>
```
