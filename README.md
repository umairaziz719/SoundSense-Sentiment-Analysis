<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Classification Models</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: #333;
        }
        header {
            background: #333;
            color: #fff;
            padding: 1em 0;
            text-align: center;
        }
        header h1 {
            margin: 0;
        }
        .container {
            width: 80%;
            margin: 2em auto;
            padding: 1em;
        }
        h2 {
            border-bottom: 2px solid #333;
            padding-bottom: 0.5em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 0.5em;
            text-align: left;
        }
        th {
            background: #f4f4f4;
        }
        code {
            background: #f4f4f4;
            padding: 0.2em;
            border-radius: 3px;
        }
        pre {
            background: #f4f4f4;
            padding: 1em;
            border-radius: 3px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <header>
        <h1>Emotion Classification Models</h1>
    </header>
    <div class="container">
        <h2 id="summary">Project Summary</h2>
        <p>This project focuses on emotion detection from audio data using various machine learning models. The primary goal is to build and evaluate different models to classify emotions based on audio features. The dataset used is the TESS (Toronto emotional speech set) dataset, which consists of audio recordings labeled with emotional states.</p>
        <p>The project includes the following models:</p>
        <ul>
            <li><strong>1D Convolutional Neural Network (CNN1D)</strong>: Used for extracting features from the audio signals and achieving high accuracy in emotion classification.</li>
            <li><strong>Long Short-Term Memory (LSTM)</strong>: A type of Recurrent Neural Network (RNN) designed to handle sequential data, such as audio signals, to capture temporal dependencies.</li>
            <li><strong>Gated Recurrent Unit (GRU)</strong>: Another RNN variant that improves training efficiency and performance by using gating mechanisms to control information flow.</li>
            <li><strong>Support Vector Machine (SVM)</strong>: A classical machine learning model used for classification tasks, which has shown competitive performance in emotion detection.</li>
        </ul>

        <h2 id="installation">Installation</h2>
        <p>To run the models and evaluate them, you'll need the following dependencies:</p>
        <ul>
            <li>Python 3.8 or higher</li>
            <li>TensorFlow 2.x</li>
            <li>scikit-learn</li>
            <li>pandas</li>
            <li>numpy</li>
        </ul>
        <p>You can install the required packages using <code>pip</code>:</p>
        <pre><code>pip install tensorflow scikit-learn pandas numpy</code></pre>

        <h2 id="dataset">Dataset</h2>
        <p>The dataset used for training and evaluating these models consists of emotional labels. The dataset should be in CSV format with features and labels. Ensure you preprocess and split the dataset into training and validation sets before running the models.</p>

        <h2 id="models">Models</h2>
        <h3>CNN1D</h3>
        <p>A 1D Convolutional Neural Network (CNN1D) is used for feature extraction from the input data. This model achieved:</p>
        <ul>
            <li><strong>Best Validation Accuracy:</strong> 99.58%</li>
            <li><strong>Final Training Accuracy:</strong> 99.66%</li>
            <li><strong>Final Validation Accuracy:</strong> 99.58%</li>
            <li><strong>Best Validation Loss:</strong> 0.1293</li>
        </ul>
        <h4>Model Architecture</h4>
        <pre><code>
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
        </code></pre>

        <h3>LSTM</h3>
        <p>A Long Short-Term Memory (LSTM) network processes sequential data. The model's architecture is as follows:</p>
        <pre><code>
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
        </code></pre>

        <h3>GRU</h3>
        <p>A Gated Recurrent Unit (GRU) model was also implemented. The model's architecture is as follows:</p>
        <pre><code>
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru_9 (GRU)                 (None, 162, 256)          198912    
                                                                 
 batch_normalization_12 (Ba  (None, 162, 256)          1024      
 tchNormalization)                                               
                                                                 
 dropout_12 (Dropout)        (None, 162, 256)          0         
                                                                 
 gru_10 (GRU)                (None, 162, 128)          148224    
                                                                 
 batch_normalization_13 (Ba  (None, 162, 128)          512       
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
        </code></pre>

        <h2 id="results">Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Final Accuracy</th>
                    <th>Final Log Loss</th>
                    <th>Precision (Overall)</th>
                    <th>Recall (Overall)</th>
                    <th>F1-Score (Overall)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>CNN1D</td>
                    <td>99.58%</td>
                    <td>0.1293</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>LSTM</td>
                    <td>77.86%</td>
                    <td>0.7029</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>GRU</td>
                    <td>87.56%</td>
                    <td>0.1687</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>SVM</td>
                    <td>96.90%</td>
                    <td>0.0956</td>
                    <td>96.92%</td>
                    <td>96.90%</td>
                    <td>96.91%</td>
                </tr>
            </tbody>
        </table>

        <h2 id="usage">Usage</h2>
        <p>To train and evaluate the models, follow these steps:</p>
        <ol>
            <li>Clone this repository:</li>
            <pre><code>git clone https://github.com/yourusername/emotion-classification.git
cd emotion-classification</code></pre>
            <li>Prepare your dataset and ensure it's in the correct format.</li>
            <li>Run the model training scripts. For example, to train the CNN1D model:</li>
            <li>Evaluate the models using the provided evaluation scripts.</li>
        </ol>

        <h2 id="license">License</h2>
        <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>
    </div>
</body>
</html>
