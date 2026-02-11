# Assignment 4: Learning Probability Density Functions using Data Only

## Author
* **Name:** Raj Fatehveer Singh Brar
* **University Roll Number:** 102317090

---

## Project Objective
The goal of this assignment is to learn an unknown probability density function (PDF) of a transformed random variable using a **Generative Adversarial Network (GAN)**. This model learns the distribution implicitly from data samples without assuming any parametric form like Gaussian or exponential distributions.

## Dataset & Transformation
The project utilizes the **India Air Quality dataset**, specifically focusing on $NO_2$ concentration as the primary feature ($x$). This feature is transformed into a new variable ($z$) using a sinusoidal function.

### Transformation Parameters
Using my university roll number ($r$), the parameters for $z = x + a_r \cdot \sin(b_r \cdot x)$ are:
* **$a_r$**: $0.5 \times (r \mod 7)$
* **$b_r$**: $0.3 \times (r \mod 5 + 1)$

## GAN Implementation
The network is designed to model the distribution of the transformed variable $z$.

* **Generator (G)**: Takes random noise from $N(0,1)$ and maps it to the distribution of $z$.
* **Discriminator (D)**: A binary classifier that distinguishes between real samples ($z$) and fake samples ($z_f$) produced by the generator.

## PDF Approximation
After training the GAN, the generator is used to produce a large number of samples ($z_f$). The final probability density $p_h(z)$ is estimated using **Kernel Density Estimation (KDE)** to visualize the distribution the model has learned.

### Generated PDF Plot
The plot below shows the final distribution learned by the GAN compared to the histogram of the generated samples.

![PDF Plot](pdf_plot.png)

## Technical Observations
* **Training Stability**: Details on the convergence of the Generator and Discriminator loss.
* **Mode Coverage**: Analysis of how well the GAN captures the various peaks in the transformed data.
* **Distribution Quality**: Comparison between the real transformed samples and the generator's output.
