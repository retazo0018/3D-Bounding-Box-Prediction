# 3D Bounding Box Prediction - Ashwin MURALI
- A deep learning model for 3D bounding box prediction given RGB, Instance Segmentation Mask, Pointcloud as inputs. 
- The code is a work in progress, intended to demonstrate the overall concept. 
- Due to limited hardware resources and time constraints, some planned features (listed in the TODO) have not yet been implemented. 
- The provided model was only trained for limited epochs and it has to be atleast trained for 50 or more epochs to produce reasonable performance.

## Data Preparation
- Resized each image, mask and pointcloud to a fixed dimension.
- Padded instance segmentation masks equal to MAX_INSTANCES.
- Padded ground truth 3d bounding boxes equal to MAX_INSTANCES.
- ![alt text](docs/data_prep.png "Data Preparation")

## Model Architecture
- The model integrates RGB and point cloud features into a unified latent representation using Transformers.
- Features from both modalities are adaptively weighted using the instance segmentation mask, which serves as an attention mask to focus on relevant regions of the input. 
- From this latent representation, a center predictor generates candidate 3D center points. Subsequently, a bounding box regressor refines these candidate centers to predict the 3D bounding box corners.
- ![alt text](docs/model_arch.png "Model Arcitecture")

## Code
- `requirements.txt` contains the packages used for this codebase.
- `main.py` loads the data from `./data` directory, prepares the data, trains and saves the model.
- `data_viz.py` visualizes a sample from the dataset.
- `test.py` evaluates the model on a test sample that was left out in training.
- `model.pt` is the trained model for limited epochs and `loss_plot.png` shows the losses over epochs visualization.
- `convert_to_onnx.py` converts the trained model into deployable low-precision ONNX format.

## ToDo's
- Add unit test
- Dockerize ?
- Include Data Augmentation to increase the training dataset.
