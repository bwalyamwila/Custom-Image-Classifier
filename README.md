# *Flower Image Classifier*
## *Overview*
This project is a deep learning-based flower image classifier that uses a pretrained neural network (e.g., VGG16) to classify images of flowers into 102 different categories. The project is divided into two parts:

Development Notebook: A Jupyter notebook that walks through the development and training of the model.

Command Line Application: A Python script that allows users to train the model and make predictions from the command line.

The model is built using PyTorch and leverages transfer learning to achieve high accuracy with minimal training time.

## Usage
Development Notebook
The Jupyter notebook (Flower_Classifier_Notebook.ipynb) contains the complete workflow for:

Loading and preprocessing the dataset.

Building and training the model.

Evaluating the model's performance.

Saving and loading the model checkpoint.

## Command Line Application
The command line application consists of two scripts:

### Training Script (train.py):

Train a new model on the dataset.

Example usage:
python train.py --data_dir data/flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu

### Prediction Script (predict.py):

Predict the class of an image using a trained model.

Example usage:
python predict.py --image_path data/flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth --top_k 5 --gpu

### Installation Prerequisites
1. Python 3.x
2. PyTorch
3. torchvision
4. NumPy
5. Matplotlib
6. PIL (Pillow)
7. Collections
8. time
9. tempfile

### Results & Performance
The model achieves a high accuracy on test data.
Predictions are visualized with confidence scores.
The command-line interface allows easy model training and inference.

### Future Improvements
Support for additional pre-trained architectures.
Hyperparameter tuning using automated search methods.
Web-based interface for real-time predictions.

License
This project is open-source and available under the MIT License.



