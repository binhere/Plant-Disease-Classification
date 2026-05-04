# Plant Disease Classification Experiments

This project is a notebook-based image classification workflow for plant disease detection. It compares a strong transformer teacher model with lightweight MobileNetV4 student models trained under different objectives, then evaluates both accuracy and deployment efficiency.

The experiments are organized as standalone Jupyter notebooks and are designed around a train/validation/test directory split of plant disease images.

## Project Structure

- `DINOv3.ipynb`: trains and evaluates a ViT model as the teacher network.
- `MNv4_CE.ipynb`: MobileNetV4 baseline trained with standard cross-entropy loss.
- `MNv4_Focal.ipynb`: MobileNetV4 experiment trained with focal loss for class-imbalance sensitivity.
- `MNv4_Distil.ipynb`: MobileNetV4 student trained with knowledge distillation from the DINOv3 teacher.
- `MNv4_Focal-Contrast.ipynb`: MobileNetV4 experiment trained with a combined focal classification loss and contrastive feature loss.

## Common Pipeline

Across the notebooks, the workflow is largely consistent:

1. Install required packages.
2. Build Albumentations-based train and validation transforms.
3. Load a plant disease dataset from a folder structure such as:

```text
root_dataset_dir/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...
  test/
    class_1/
    class_2/
    ...
```

4. Train with mixed precision, AdamW, warmup, cosine annealing, checkpointing, and early stopping.
5. Evaluate on the test split with:
   - classification report
   - macro F1, precision, recall, accuracy
   - normalized confusion matrix
   - sample prediction visualization
6. Benchmark the trained model for latency, throughput, model size, and dtype behavior.
7. Export and benchmark an ONNX version for CPU inference.

## Models and Experiment Goals

### 1. DINOv3 Teacher

The teacher notebook uses a pretrained DINOv3 Vision Transformer backbone from `timm/vit_small_patch16_dinov3_qkvb.lvd1689m` and fine-tunes it for plant disease classification. This notebook serves two purposes:

- establish a higher-capacity reference model
- provide the teacher checkpoint used by the distillation notebook

### 2. MobileNetV4 Cross-Entropy Baseline

The baseline student model uses `mobilenetv4_conv_small_050.e3000_r224_in1k` with a configurable classifier head. This notebook is the main lightweight reference for deployment-oriented classification.

### 3. MobileNetV4 Focal Loss

This variant keeps the MobileNetV4 backbone but replaces standard cross-entropy with focal loss. It is useful when class imbalance or hard-example emphasis matters.

### 4. MobileNetV4 Focal + Contrastive Loss

This notebook extends the MobileNetV4 setup with a combined objective:

- focal loss on class logits for classification
- contrastive loss on a learned projection head to shape feature embeddings

During training, the model returns both:

- classification logits from the classifier head
- projected features from a separate projector network

The idea is to improve class discrimination not only at the output layer, but also in the embedding space, so samples from the same class are encouraged to cluster while different classes are pushed apart. This makes the notebook a representation-learning variant rather than only a loss replacement.

### 5. MobileNetV4 Knowledge Distillation

This notebook trains the same MobileNetV4 student while matching both:

- hard labels from the dataset
- soft targets from the DINOv3 teacher

The goal is to transfer some of the teacher's representational quality into a smaller inference-friendly model.

## Training Details

The notebooks include the following training features:

- configurable frozen or fine-tuned backbone modes
- optional linear or MLP classifier heads
- optional projection head for the focal-plus-contrastive experiment
- AdamW optimizer with separate learning rates for backbone and classifier
- linear warmup followed by cosine annealing
- AMP training on CUDA
- checkpoint saving for best validation F1
- crash checkpoint saving for interrupted runs

## Data Augmentation

The training notebooks use Albumentations for image preprocessing and augmentation, including:

- resize
- CLAHE
- horizontal and vertical flips
- random 90-degree rotation
- shift, scale, and rotate augmentation
- color jitter
- Gaussian noise
- Gaussian blur
- normalization and tensor conversion

## Evaluation and Benchmarking

Each main notebook includes utilities to inspect both model quality and deployment cost.

Evaluation outputs typically include:

- test loss
- macro F1, precision, recall, and accuracy
- per-class report
- confusion matrix
- correct and incorrect sample predictions

Benchmarking utilities compare:

- float32 and float16 inference
- latency per image
- throughput in images per second
- serialized model size
- ONNX CPU inference performance

## Dependencies

The notebooks install or use the following main packages:

- `torch`
- `torchvision`
- `timm`
- `albumentations`
- `thop`
- `onnx`
- `onnxruntime`
- `onnxscript`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `Pillow`
- `tqdm`

## How To Use

Recommended experiment order:

1. Run `DINOv3.ipynb` to train the teacher and save the best checkpoint.
2. Run `MNv4_CE.ipynb` to establish the MobileNetV4 baseline.
3. Run `MNv4_Focal.ipynb` to compare focal loss against the baseline.
4. Run `MNv4_Focal-Contrast.ipynb` to test whether adding contrastive embedding supervision improves the MobileNetV4 representation.
5. Run `MNv4_Distil.ipynb` to train a distilled MobileNetV4 student using the saved teacher checkpoint.
6. Compare final test metrics and benchmark tables across notebooks.