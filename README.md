```markdown
# SoundSense:Emotion Detection

## Project Summary

This project focuses on emotion detection from audio data using various machine learning models. The goal is to classify emotions based on audio features from the TESS (Toronto Emotional Speech Set) dataset, which consists of emotional speech recordings.

## Installation

To get started with this project, you'll need the following dependencies:

- Python 3.8 or higher
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy

You can install the required packages using pip:

```bash
pip install tensorflow scikit-learn pandas numpy
```

## Dataset

The dataset used for training and evaluating the models is the TESS dataset, which contains emotional labels. Ensure you preprocess the dataset and split it into training and validation sets before running the models. The dataset should be in CSV format with features and labels.

## Models

### 1D Convolutional Neural Network (CNN1D)

The CNN1D model is used for feature extraction from audio signals. The architecture is as follows:

```
Model: "sequential_11"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d_10 (Conv1D)          (None, 162, 512)          3072      
                                                                 
 batch_normalization_49 (Ba  (None, 162, 512)          2048      
 tchNormalization)                                               
                                                                 
 max_pooling1d_10 (MaxPooli  (None, 81, 512)           0         
 ng1D)                                                           
                                                                 
 dropout_49 (Dropout)        (None, 81, 512)           0         
                                                                 
 conv1d_11 (Conv1D)          (None, 81, 256)           655616    
                                                                 
 batch_normalization_50 (Ba  (None, 81, 256)           1024      
 tchNormalization)                                               
                                                                 
 max_pooling1d_11 (MaxPooli  (None, 41, 256)           0         
 ng1D)                                                           
                                                                 
 dropout_50 (Dropout)        (None, 41, 256)           0         
                                                                 
 conv1d_12 (Conv1D)          (None, 41, 128)           163968    
                                                                 
 batch_normalization_51 (Ba  (None, 41, 128)           512       
 tchNormalization)                                               
                                                                 
 max_pooling1d_12 (MaxPooli  (None, 21, 128)           0         
 ng1D)                                                           
                                                                 
 dropout_51 (Dropout)        (None, 21, 128)           0         
                                                                 
 conv1d_13 (Conv1D)          (None, 21, 128)           49280     
                                                                 
 batch_normalization_52 (Ba  (None, 21, 128)           512       
 tchNormalization)                                               
                                                                 
 max_pooling1d_13 (MaxPooli  (None, 11, 128)           0         
 ng1D)                                                           
                                                                 
 dropout_52 (Dropout)        (None, 11, 128)           0         
                                                                 
 conv1d_14 (Conv1D)          (None, 11, 64)            24640     
                                                                 
 batch_normalization_53 (Ba  (None, 11, 64)            256       
 tchNormalization)                                               
                                                                 
 max_pooling1d_14 (MaxPooli  (None, 6, 64)             0         
 ng1D)                                                           
                                                                 
 dropout_53 (Dropout)        (None, 6, 64)             0         
                                                                 
 flatten_2 (Flatten)         (None, 384)               0         
                                                                 
 dense_22 (Dense)            (None, 256)               98560     
                                                                 
 batch_normalization_54 (Ba  (None, 256)               1024      
 tchNormalization)                                               
                                                                 
 dropout_54 (Dropout)        (None, 256)               0         
                                                                 
 dense_23 (Dense)            (None, 7)                 1799      
                                                                 
=================================================================
Total params: 1002311 (3.82 MB)
Trainable params: 999623 (3.81 MB)
Non-trainable params: 2688 (10.50 KB)
_________________________________________________________________
```

### Long Short-Term Memory (LSTM)

The LSTM model processes sequential data to capture temporal dependencies. The architecture is:

```
Model: "sequential_5"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm_20 (LSTM)                       │ (None, 162, 128)            │          66,560 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_25               │ (None, 162, 128)            │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_25 (Dropout)                 │ (None, 162, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_21 (LSTM)                       │ (None, 162, 64)             │          49,408 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_26               │ (None, 162, 64)             │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_26 (Dropout)                 │ (None, 162, 64)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_22 (LSTM)                       │ (None, 162, 32)             │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_27               │ (None, 162, 32)             │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_27 (Dropout)                 │ (None, 162, 32)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_23 (LSTM)                       │ (None, 32)                  │           8,320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_28               │ (None, 32)                  │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_28 (Dropout)                 │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_10 (Dense)                     │ (None, 128)                 │           4,224 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_29               │ (None, 128)                 │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_29 (Dropout)                 │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_11 (Dense)                     │ (None, 7)                   │             903 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 143,367 (560.03 KB)
 Trainable params: 142,599 (557.03 KB)
 Non-trainable params: 768 (3.00 KB)
```

### Gated Recurrent Unit (GRU)

The GRU model is an efficient RNN variant. The architecture is:

```
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru_9 (GRU)                 (None, 162, 256)          198912    
                                                                 
 batch_normalization_12 (Ba  (None, 162, 256)          1024      
 tchNormalization)                                               
                                                                 
 dropout_12 (Dropout)        (None, 162, 256)          0         
                                                                 
 gru_10 (GRU)                (None, 162, 128)          148224    
                                                                 
 batch_normalization_13 (Ba  (None, 162, 128

)          512       
 tchNormalization)                                               
                                                                 
 dropout_13 (Dropout)        (None, 162, 128)          0         
                                                                 
 gru_11 (GRU)                (None, 64)                37248     
                                                                 
 batch_normalization_14 (Ba  (None, 64)                256       
 tchNormalization)                                               
                                                                 
 dropout_14 (Dropout)        (None, 64)                0         
                                                                 
 dense_6 (Dense)             (None, 512)               33280     
                                                                 
 batch_normalization_15 (Ba  (None, 512)               2048      
 tchNormalization)                                               
                                                                 
 dropout_15 (Dropout)        (None, 512)               0         
                                                                 
 dense_7 (Dense)             (None, 7)                 3591      
                                                                 
=================================================================
Total params: 425095 (1.62 MB)
Trainable params: 423175 (1.61 MB)
Non-trainable params: 1920 (7.50 KB)
_________________________________________________________________
```

## Results

| Model | Final Accuracy | Final Log Loss | Precision (Overall) | Recall (Overall) | F1-Score (Overall) |
|-------|----------------|----------------|---------------------|------------------|--------------------|
| CNN1D | 99.58%         | 0.1293         | -                   | -                | -                  |
| LSTM  | 77.86%         | 0.7029         | -                   | -                | -                  |
| GRU   | 87.56%         | 0.1687         | -                   | -                | -                  |
| SVM   | 96.90%         | 0.0956         | 96.92%              | 96.90%           | 96.91%             |

## Usage

To train and evaluate the models, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/umairaziz719/SoundSense-Sentiment-Analysis.git
    ```

2. Prepare your dataset and ensure it's in the correct format.

3. Run the model training scripts.

4. Evaluate the models using the provided evaluation scripts.

## License

This project is licensed under the Creative Common License.
```

### Key Sections:
- **Project Summary**: Briefly describes the project's goal.
- **Installation**: Lists dependencies and installation commands.
- **Dataset**: Instructions for preparing the dataset.
- **Models**: Details the architectures of CNN1D, LSTM, and GRU models.
- **Results**: A table summarizing the performance of each model.
- **Usage**: Instructions for cloning the repo, preparing the dataset, and running training scripts.
- **License**: Licensing information.

