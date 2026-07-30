# ModelForge

<div align="center">

![ModelForge Logo](./static/architecture.jpg)

**Model Clustering and Consolidation System for Machine Learning Models**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📋 Overview

ModelForge is an advanced system for clustering and consolidating machine learning models. It efficiently reduces a large collection of models into a smaller set of representative models while preserving performance characteristics.

## ✨ Key Features

- **Model Clustering** - Group similar models based on performance metrics and characteristics
- **Model Consolidation** - Generate a smaller, representative set of models from the original collection
- **Performance Evaluation** - Comprehensively evaluate consolidated models against the original set
- **Visualization Tools** - Analyze model similarities and differences through intuitive visualizations

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry (dependency management)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/modelforge.git
cd modelforge
poetry install
```

### Quick Usage

See our [demo notebook](docs/demo.ipynb) for a minimal example of how to use ModelForge.

## 📚 Documentation

- [Using Your Own Datasets](docs/own_dataset.md) - Learn how to create and format your own datasets
- [Evaluation Datasets](docs/datasets.md) - Details on datasets used in ModelForge evaluation
- [Reproducibility](docs/reproducibility.md) - Instructions to reproduce our research results
- [Visualization & Plots](docs/plots.md) - Detailed explanation of visualization options and additional results

## 🔬 Research & Results

Our research demonstrates significant efficiency gains when using ModelForge for model consolidation while maintaining performance thresholds. See our [paper](link-to-paper) for complete details.

## 📝 Citation

If you use ModelForge in your research, please cite our paper:

```bibtex
@inproceedings{Siepe2025,
  author={Siepe, Florian and Prinz, Thomas Peter and Papenbrock, Thorsten},
  booktitle={2025 IEEE 12th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={ModelForge: A Metric Approach to Machine Learning Model Consolidation}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/DSAA65442.2025.11247992}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

