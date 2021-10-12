# NeLLoC

This is a demo to do lossless compression with NeLLoC (with Arithmetic Coding) on CPU!

The model stucture is a local PixelCNN with one CNN layer followed by two 1x1 conv nets (0 ResNet). We use a discretized logitic-uniform mixture distribution as emission distribution.  The model size is only *105 KB* (size of PyTorch .pt file).

### Performance of NeLLoC on CPU (MacBook Air 2020, M1 chip)
Here we list the compression performance, all the experiments tested using a single CPU (M1 chip, MacBook Air 2020). See demo.ipynb file for details.

| NeLLoC   (105 KB)     |    SVHN        | CIFAR  |
| ------------- |:-------------:| -----:|
| BPD      | 2.38 | 3.64 |
| Compression time (s)     | 0.376      |   0.434 |
| Decompression time (s)      | 0.401      |  0.471 |

### Comparison to HiLLoCon CPU (Razor Blade 15, i7-10750H)
Since we didn't manage to run HiLLoC (jax implementation) on macbook air m1 chip, therefore, we run the comparisons on a Razor Blade 15 with CPU i7-10750H.  We test two models for compressing a single image (see the file hilloc_comparisons for details).


| Method |    BPD  |loading time (s) | Compression time (s)| Decompression time (s) | CPU memory usage |
| ------------- |:-------------:| -----:| -----:|-----:|-----:|
| NeLLoC      | 3.99 | 0.0013| 0.71| 0.82|  0.126 MB|
| HiLLoC     | 4.0  |0.682    |   5.21 | 0.35 |  317.0 MB |










