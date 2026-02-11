# India Air Quality - PDF Estimation via GANs

## Project Objective
The goal of this project is to learn an unknown probability density function (PDF) of a transformed random variable using a Generative Adversarial Network (GAN). Instead of assuming a standard distribution (like Gaussian or Exponential), the GAN learns the distribution implicitly from the data samples provided.

## Transformation Logic
The dataset uses NO2 concentration as the primary feature ($x$). This feature is transformed into a new variable ($z$) using a specific sinusoidal transformation:

$$z = x + a_r \cdot \sin(b_r \cdot x)$$

Where the parameters $a_r$ and $b_r$ are derived from the university roll number:
* $a_r = 0.5 \times (\text{roll\_mod\_7})$
* $b_r = 0.3 \times (\text{roll\_mod\_5} + 1)$

## GAN Architecture
The model consists of two competing networks:

### 1. The Generator
* **Input**: Random noise following $N(0,1)$.
* **Structure**: Fully connected layers with ReLU activations.
* **Role**: To map the latent noise space to the distribution of the transformed variable $z$.

### 2. The Discriminator
* **Input**: Real samples ($z$) or fake samples ($z_f$).
* **Structure**: Fully connected layers with LeakyReLU activations and a Sigmoid output.
* **Role**: To distinguish between the actual transformed data and the samples produced by the generator.



## Methodology
1.  **Preprocessing**: Cleaning the India Air Quality dataset and scaling the NO2 features.
2.  **Transformation**: Applying the roll-number-based trigger function to generate the target variable $z$.
3.  **Adversarial Training**: Training both networks simultaneously using Binary Cross Entropy (BCE) loss.
4.  **PDF Approximation**: After training, 10,000 samples are generated from the Generator. The final PDF is estimated using Kernel Density Estimation (KDE) and compared against the sample histogram.

## Requirements
* Python 3.x
* PyTorch
* Pandas / NumPy
* Matplotlib / Scipy

## Results
The trained generator successfully captures the modes and variance of the transformed distribution. The final PDF plot (`pdf_plot.png`) visualizes the density learned by the model compared to the actual data distribution.
