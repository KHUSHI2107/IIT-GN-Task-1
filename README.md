# IIT-GN-Task-1
Task-1 for summer research internship 2022 at IIT Gandhinagar under the supervision of **Prof. Nipun Batra sir**. 
**Animate bivariate normal distribution-**

![unnamed](https://user-images.githubusercontent.com/65617775/162606334-7f3496e5-a1a3-4619-be81-861680c96188.png)

* Reproduce the above figure showing samples from bivariate normal with marginal PDFs from scratch using JAX and matplotlib.
* Add interactivity to the figure by adding sliders with ipywidgets. You should be able to vary the parameters of bivariate normal distribution (mean and        covariance matrix) using ipywidgets.

**GETTING HANDS-ON WITH JAX-**

The aim is to introduce you with [JAX](https://github.com/google/jax) and complete the task. 

JAX ecosystem is becoming an increasingly popular alternative to PyTorch and TensorFlow

![jax_logo](https://user-images.githubusercontent.com/65617775/162607900-5c763001-1b6e-4af5-8842-dc82e467f657.png)

Boiled down, JAX is python's numpy with automatic differentiation and optimized to run on GPU. The seamless translation between writing numpy and writing in JAX has made JAX popular with machine learning practitioners.
JAX offers four main function transformations that make it efficient to use when executing deep learning workloads. At its core, JAX is an extensible system for transforming numerical functions. Here are four transformations of primary interest: grad, jit, vmap, and pmap.

* grad - automatically differentiates a function for backpropagation. You can take grad to any derivative order.

* jit - auto-optimizes your functions to run their operations efficiently. Can also be used as a function decorator. You can use XLA to compile your functions end-to-end with jit, used either as an @jit decorator or as a higher-order function.

* vmap - maps a function across dimensions. Means that you don't have to keep track of dimensions as carefully when passing a batch through, for example. vmap is the vectorizing map. It has the familiar semantics of mapping a function along array axes, but instead of keeping the loop on the outside, it pushes the loop down into a function’s primitive operations for better performance.

* pmap - maps processes across multiple processors, like multi-GPU. For parallel programming of multiple accelerators, like multiple GPUs, use pmap. With pmap you write single-program multiple-data (SPMD) programs, including fast parallel collective communication operations. Applying pmap will mean that the function you write is compiled by XLA (similarly to jit), then replicated and executed in parallel across devices.






