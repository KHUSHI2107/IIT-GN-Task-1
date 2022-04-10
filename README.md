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

**Installation-**

JAX is written in pure Python, but it depends on XLA, which needs to be installed as the jaxlib package. Use the following instructions to install a binary package with pip, or to build JAX from source.

We support installing or building jaxlib on Linux (Ubuntu 16.04 or later) and macOS (10.12 or later) platforms.
Windows users can use JAX on CPU and GPU via the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about). There is some initial native Windows support, but since it is still somewhat immature, there are no binary releases and it must be [built from source](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-jaxlib-from-source-on-windows).

**pip installation: CPU**

To install a CPU-only version of JAX, which might be useful for doing local development on a laptop, you can run

~~~pip install --upgrade pip
   pip install --upgrade "jax[cpu]" 
~~~

On Linux, it is often necessary to first update pip to a version that supports manylinux2010 wheels.

**pip installation: GPU (CUDA)**

If you want to install JAX with both CPU and NVidia GPU support, you must first install [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuDNN](https://developer.nvidia.com/CUDNN), if they have not already been installed. Unlike some other popular deep learning systems, JAX does not bundle CUDA or CuDNN as part of the pip package.

JAX provides pre-built CUDA-compatible wheels for Linux only, with CUDA 11.1 or newer, and CuDNN 8.0.5 or newer. Other combinations of operating system, CUDA, and CuDNN are possible, but require building from source.

* CUDA 11.1 or newer is required.
You may be able to use older CUDA versions if you build from source, but there are known bugs in CUDA in all CUDA versions older than 11.1, so we do not ship prebuilt binaries for older CUDA versions.
* The supported cuDNN versions for the prebuilt wheels are:
cuDNN 8.2 or newer. We recommend using the cuDNN 8.2 wheel if your cuDNN installation is new enough, since it supports additional functionality.
cuDNN 8.0.5 or newer.
* You must use an NVidia driver version that is at least as new as your [CUDA toolkit's corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions). For example, if you have CUDA 11.4 update 4 installed, you must use NVidia driver 470.82.01 or newer if on Linux. This is a strict requirement that exists because JAX relies on JIT-compiling code; older drivers may lead to failures.
If you need to use an newer CUDA toolkit with an older driver, for example on a cluster where you cannot update the NVidia driver easily, you may be able to use the [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/) that NVidia provides for this purpose.

Next, run

~~~
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
~~~

The jaxlib version must correspond to the version of the existing CUDA installation you want to use. You can specify a particular CUDA and CuDNN version for jaxlib explicitly:

~~~pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
~~~

You can find your CUDA version with the command:

~~~
nvcc --version
~~~

Some GPU functionality expects the CUDA installation to be at /usr/local/cuda-X.X, where X.X should be replaced with the CUDA version number (e.g. cuda-11.1). If CUDA is installed elsewhere on your system, you can either create a symlink:

~~~
sudo ln -s /path/to/cuda /usr/local/cuda-X.X
~~~

**pip installation: Google Cloud TPU-**

JAX also provides pre-built wheels for [Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm). To install JAX along with appropriate versions of jaxlib and libtpu, you can run the following in your cloud TPU VM:

~~~
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
~~~

**pip installation: Colab TPU**

Colab TPU runtimes come with JAX pre-installed, but before importing JAX you must run the following code to initialize the TPU:

~~~
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
~~~

Colab TPU runtimes use an older TPU architecture than Cloud TPU VMs, so installing jax[tpu] should be avoided on Colab. If for any reason you would like to update the jax & jaxlib libraries on a Colab TPU runtime, follow the CPU instructions above (i.e. install jax[cpu]).


**NORMAL DISTRIBUTION**-

The normal distribution was first discovered by De-Moivre (an English mathematician) in 1733. De-Moivre obtained this continuous distribution as a limiting case of the binomial distribution and applied it to the problems of game of chance. It waw credited to Gauss (1809) who used the normal curve to describe the theory of accidental errors of measurements involved in the calculation of orbits of heavenly bodies.
Throughout two continuous centuries (18th and 19th) many efforts were made to develop a normal model as the underlying law which may govern all continuous random variables. That is why the name 'normal.

**Definition.** A random variable X is said to have a normal distribution with parameters m and o2 if its density function is given by the probability law :


![normal_distribution](https://user-images.githubusercontent.com/65617775/162609850-20465b44-beda-4cd0-be42-3ca06d9e3164.svg)

**f(x)**	=	probability density function,  
**σ** =	standard deviation,  
**μ**	=	mean

- ∞ < x < ∞ , σ > 0 , - ∞ < m < ∞

where mn is called '**mean**' and o is called '**variance**'.


**Bivariate Gaussian distribution**

The multivariate Gaussian distribution of an n-dimensional vector x=(x1,x2,⋯,xn) may be written

p(x;μ,Σ)=1(2π)n|Σ|−−−−−−−√exp(−12(x−μ)TΣ−1(x−μ)),
where μ is the n-dimensional mean vector and Σ is the n×n covariance matrix.

To visualize the magnitude of p(x;μ,Σ) as a function of all the n dimensions requires a plot in n+1 dimensions, so visualizing this distribution for n>2 is tricky. The code below calculates and visualizes the case of n=2, the bivariate Gaussian distribution.
