# Diffeomorphic demons in TensorFlow

> This is a course project in *Medical Image Analysis* on diffeomorphic demons and is implemented in TensorFlow. The lecture note for illustration and presentation is also included.

## Getting Started

### Project structure

The project contains 2D and 3D versions of Diffeomorphic demons and a dedicated version for circle-to-C registration.

```
Diffeomorphic-demons
|-- src      
|    |-- circle2C_reg.py                   # circle-to-C registration
|    |-- compare_metrics.py                # compare metrics from different settings
|    |-- crop_ROI.py                       # image preprocessing
|    |-- trainer_2d.py                     # 2D version of Diffeomorphic-demons
|    |-- trainer_3d.py                     # 3D version of Diffeomorphic-demons
|    |-- transformer.py                    # spatial transformer modules adapted from VoxelMorph
|    |-- utils.py                          # utility functions for preprocessing and evaluation
```

### Usage

Registering a 3D fixed image with a 3D moving image is achieved by:

```
python trainer_3d.py
--demons_type diffeomorphic               # the transformation model for demons
--demons_force symmetric                  # demons force type
--regularizer fluid                       # regularization type for the displacement field
--normalization z-score                   # image normalization type during preprocessing
--max_length 2.                           # the maximum step length
--exp_steps 8                             # the number of exponential steps for the vector field
--training_iters 30                    
--display_steps 5
--cuda_device -1                          # use CPU only
```

## Acknowledgement

Some parts of the code were adapted from [VoxelMorph](https://github.com/voxelmorph/voxelmorph/tree/master), which is for deep-learning-based medical image registration.

## Citation

If you found the repository useful, please cite the lecture note as below:

```bibtex
@misc{Luo2019DiffeomorphicDemons,
  title={Medical Image Analysis, Diffeomorphic Demons},
  author={Xinzhe Luo},
  year={2019}
}
```

## Contact

For any questions or problems please [open an issue](https://github.com/xzluo97/Diffeomorphic-demons/issues/new) on GitHub.
