# 3D Bounding Box Prediction - Ashwin MURALI
- A framework to train a transformer-based fusion model for 3D bounding box prediction given RGB, Instance Segmentation Mask and Pointcloud as inputs. 
- The framework produces an optimized ONNX model suitable for deployments.

## Getting-Started
- Clone the repository
- Install Dependencies: Run `pip install -r requirements.txt`
- Run unit tests: `pytest -v`
- Train the Model: `python main.py --data_dir <PATH TO DATA DIR> --epochs <EPOCHS> --batch_size <BATCH_SIZE>`
- Optimize the Model as ONNX: `python convert_to_onnx.py`
- Infer the Model: `python run_inference.py`

## Implementation
| ![alt text](docs/data_prep.png "Data Preparation") |
|:--:|
| *Figure 1: System Architecture* |

### Data Preprocessing
To ensure consistency across samples with varying shapes and sizes, the following preprocessing steps were applied:

- **Resizing**: Each image, mask, and point cloud was resized to a FIXED_DIMENSION. FIXED_DIMENSION comprises of a height (H_FIX) and a width (W_FIX).
- **Instance Mask Padding**: Instance segmentation masks were padded to match a fixed maximum number of instances (MAX_INSTANCES).
- **3D Bounding Box Padding**: Ground truth 3D bounding boxes were padded to also match MAX_INSTANCES.
- MAX_INSTANCES is a hyperparameter and set to `25`.
- FIXED_DIMENSION is hyperparameter and set to `640, 640`

### Model Architecture
- The model integrates RGB and point cloud features into a unified latent representation using Transformers.
- Features from both modalities are adaptively weighted using the instance segmentation mask before the fusion, which serves as an attention mask to focus on relevant regions of the input. 
- From this latent representation, a center predictor generates candidate 3D center points. Subsequently, a bounding box regressor refines these candidate centers to predict the 3D bounding box corners.

| ![alt text](docs/model_arch.png "Model Architecture") |
|:--:|
| *Figure 2: Model Architecture* |

### Hybrid 3D Bbox Loss Function
This custom loss function is designed to improve the accuracy of 3D bounding box prediction by combining two complementary terms. L2 loss helps with numerical stability. Chamfer distance improves geometric alignment and is more tolerant to corner ordering issues. Together, they guide the network to learn accurate and perceptually correct 3D box shapes.

- **L2 Loss (Mean Squared Error)**: Measures the direct difference between predicted and ground truth 3D box corner coordinates.
- **Chamfer Distance**: Computes the average closest-point distance between the predicted and ground truth corner sets.

## Sample Results
Training and evaluation logs are saved in log.txt. The loss_plot.png file visualizes the loss over epochs, providing insight into the model's training behavior.
| ![alt text](loss_plot.png "Loss Plot") |
|:--:|
| *Figure 3: Model Training* |

## Codebase
- `requirements.txt` contains the packages used for this codebase.
- `run_unit_tests.py` runs unit tests to test the framework.
- `data_viz.py` visualizes a sample from the dataset.
- `main.py` loads the data from `./data` directory, prepares the data, trains and saves the model.
- `convert_to_onnx.py` converts the trained model into deployable low-precision ONNX format.
- `run_inference.py` evaluates the model on a test sample that was left out in training.
- `model.pt` is the trained model, `model.onnx` is the optimized model and `loss_plot.png` shows the losses over epochs visualization.

# Future Work
- Train for longer epochs to better capture learning dynamics and reduce underfitting.
- Enhance model generalization and accuracy by augmenting the RGB images with robust data augmentation techniques such as geometric transformations, color jittering, and Gaussian noise ; while ensuring consistency across the corresponding masks and point clouds.


# Acknowledgments
- [Stack Overflow](https://stackoverflow.com/questions) - Helped resolve aand debug technical issues in the implementation.
- [ChatGPT](https://chatgpt.com/) – Assisted with refining README documentation and docstrings in code.

# Future Work
- Train for longer epochs to better capture learning dynamics and reduce underfitting.
- Improve model generalization and accuracy by expanding the dataset using robust data augmentation techniques (e.g., geometric transforms, color jittering, or simulated noise).
