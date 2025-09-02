# Conditional GAN for Text-to-Image Synthesis

A PyTorch implementation of a Conditional Generative Adversarial Network (cGAN) that generates realistic flower images from textual descriptions using the Oxford-102 flower dataset.

## Overview

This project demonstrates an advanced conditional GAN architecture that combines a Source Encoder with a Target Generator to synthesize high-quality 128x128 RGB images conditioned on text embeddings. Unlike traditional GANs that rely solely on random noise, this implementation leverages both visual features extracted from real images and semantic information from text descriptions to generate contextually relevant and diverse flower images.

## Architecture

The system consists of three main components:

### 1. Source Encoder
- **Purpose**: Extracts meaningful 1024-dimensional feature representations from real images
- **Architecture**: Hybrid CNN combining ResNet residual blocks with Inception modules
- **Flow**: 128x128x3 → Multiple residual layers → Inception blocks → Global average pooling → 1024D features
- **Innovation**: Multi-scale feature extraction through parallel convolutional branches

### 2. Target Generator
- **Purpose**: Synthesizes realistic images from concatenated features (source + text embeddings)
- **Input**: 2048-dimensional latent vector (1024D source features + 1024D text embeddings)
- **Architecture**: Progressive upsampling using transposed convolutions
- **Flow**: 2048D vector → 8x8 → 16x16 → 32x32 → 64x64 → 128x128x3
- **Output**: RGB images with Tanh activation (values in [-1, 1])

### 3. Discriminator
- **Purpose**: Distinguishes between real and generated images for adversarial training
- **Architecture**: Convolutional network with progressive downsampling
- **Flow**: 128x128x3 → 64x64 → 32x32 → 16x16 → 8x8 → 1x1 → Scalar output

## Key Features

- **Hybrid CNN Architecture**: Combines ResNet residual connections with Inception multi-scale feature extraction
- **Multi-Loss Training**: Employs both Binary Cross-Entropy and Mean Squared Error losses for enhanced training stability
- **End-to-End Learning**: Joint optimization of source encoder and generator for optimal visual-semantic mapping
- **Comprehensive Evaluation**: Includes quantitative analysis via t-SNE visualization and qualitative assessment through image generation
- **Memory Efficient**: Optimized for standard GPU training with reasonable computational requirements

## Dataset

**Oxford-102 Flower Dataset**
- **Source**: University of Oxford Visual Geometry Group
- **Content**: 8,189 flower images across 102 categories
- **Text Embeddings**: Pre-computed 1024-dimensional semantic representations
- **Split**: 20 classes for training, 5 classes for testing (random selection with seed=0)
- **Preprocessing**: Images resized to 128x128 pixels, normalized to [0,1] range

## Requirements

### Dependencies
```
torch >= 1.9.0
torchvision >= 0.10.0
torchfile
numpy
scipy
matplotlib
scikit-learn
```

### Hardware Requirements
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (recommended)
- **CPU**: Multi-core processor for data loading
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for dataset and model checkpoints

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Conditional-GAN-Image-Synthesis.git
cd Conditional-GAN-Image-Synthesis
```

2. **Install dependencies**
```bash
pip install torch torchvision torchfile numpy scipy matplotlib scikit-learn
```

3. **Download dataset** (optional - can be done within notebook)
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz
```

## Usage

### Quick Start
Open and run the Jupyter notebook:
```bash
jupyter notebook ConditionalGAN.ipynb
```

### Training
The notebook provides a complete training pipeline:

1. **Data Preparation**: Automated dataset loading and preprocessing
2. **Model Initialization**: Instantiation of encoder, generator, and discriminator
3. **Training Loop**: 200 epochs with progress visualization every 20 epochs
4. **Loss Monitoring**: Real-time loss curve plotting for training stability analysis

### Evaluation
The project includes comprehensive evaluation metrics:

1. **Image Generation**: 5x5 grid of generated images per test class
2. **Feature Analysis**: 3D t-SNE visualization of learned representations
3. **Model Analysis**: Parameter count and memory footprint assessment

## Implementation Details

### Training Configuration
- **Optimizer**: Adam with learning rate 0.0002, betas (0.5, 0.999)
- **Batch Size**: 32
- **Epochs**: 200
- **Loss Functions**: BCE + MSE for enhanced stability
- **Visualization**: Generated samples every 20 epochs

### Model Parameters
- **Source Encoder**: ~25M parameters
- **Generator**: ~12M parameters (constraint: half of encoder)
- **Discriminator**: ~8M parameters
- **Total Training Parameters**: ~45M

### Performance Metrics
- **Training Time**: ~1 hour on modern GPU (GTX 1080 Ti or better)
- **Memory Usage**: ~4GB GPU VRAM during training
- **Convergence**: Stable training with balanced discriminator-generator dynamics

## Results

### Generated Images
The model produces high-quality 128x128 flower images with:
- **Visual Fidelity**: Realistic textures, colors, and structures
- **Semantic Consistency**: Generated images align with input text descriptions
- **Diversity**: Varied outputs for similar text inputs
- **Class Separation**: Clear distinction between different flower types

### Feature Learning
t-SNE analysis demonstrates:
- **Clustering**: Clear separation of different flower classes in feature space
- **Generalization**: Effective feature extraction on unseen test classes
- **Semantic Structure**: Meaningful organization of visual representations

## Project Structure

```
Conditional-GAN-Image-Synthesis/
├── ConditionalGAN.ipynb          # Main implementation notebook
├── README.md                     # Project documentation
├── 102flowers/                   # Dataset directory (created after download)
│   └── jpg/                      # Raw images
├── flowers_icml/                 # Text embeddings (created after download)
│   └── class_*/                  # Embeddings per class
├── train/                        # Training data (created during execution)
│   ├── images/                   # Processed training images
│   └── embeddings/               # Processed training embeddings
└── test/                         # Test data (created during execution)
    ├── images/                   # Processed test images
    └── embeddings/               # Processed test embeddings
```

## Technical Contributions

### Novel Architecture Design
- **Hybrid Feature Extraction**: Combination of ResNet and Inception architectures for multi-scale feature learning
- **Conditional Generation**: Integration of visual and textual features for controlled image synthesis
- **Memory Optimization**: Efficient architecture design suitable for standard GPU configurations

### Training Innovations
- **Multi-Loss Strategy**: Combined BCE and MSE losses for improved training stability
- **Progressive Visualization**: Regular monitoring of generation quality during training
- **Balanced Optimization**: Careful tuning of discriminator-generator learning dynamics

## Applications

This implementation can be adapted for:
- **Art Generation**: Creating artistic flower illustrations from descriptions
- **Dataset Augmentation**: Generating synthetic training data for flower classification
- **Educational Tools**: Demonstrating conditional GAN concepts and text-to-image synthesis
- **Research Platform**: Base implementation for advanced conditional generation research

## Future Enhancements

Potential improvements and extensions:
- **Multi-Resolution Training**: Progressive growing for higher resolution outputs
- **Attention Mechanisms**: Visual attention for better text-image alignment
- **Advanced Losses**: Perceptual and feature matching losses for improved quality
- **Interactive Interface**: Web-based demo for real-time generation

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{conditional-gan-text-to-image,
  title={Conditional GAN for Text-to-Image Synthesis},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/Conditional-GAN-Image-Synthesis}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Oxford Visual Geometry Group for the Oxford-102 flower dataset
- PyTorch team for the deep learning framework
- Original GAN and conditional GAN research communities

## Contact

For questions, suggestions, or collaborations:
- **Email**: your.email@domain.com
- **GitHub**: [yourusername](https://github.com/yourusername)
- **LinkedIn**: [yourprofile](https://linkedin.com/in/yourprofile)

---

**Note**: This implementation is designed for educational and research purposes. The architecture and training procedures can be adapted for other text-to-image synthesis tasks with appropriate dataset modifications.