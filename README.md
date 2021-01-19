# LBP Descriptor CUDA
Project for the Parallel Computing exam at the University of Florence.

The sequential and first parallel implementations for the same project can be found here: [LBP descriptor](https://github.com/matpetrone/LBP_Descriptor).

Local binary patterns (LBP) is a type of visual descriptor used for classification in computer vision. It has since been found to be a powerful feature for texture classification.

<p align="center">
  <img src="https://github.com/matpetrone/LBP_Descriptor_CUDA/res/images/readme_img/ldp_alg" width="600">
</p>

### Directories Layout

```bash
├── src                       # Source files
│   ├── ...
├── res                       # Resources files
│   ├── histograms            # Histograms csv and plot
│   ├── images                # Images 
│   │   ├── ppm               # Images for experiments in ppm format
│   ├── plots                 # Plots
│   ├── results               # Experiments results
├── docs                      # Documentation files
│   ├── LBP_descriptor_paper  # LBP analysis and experiments results 
│   ├── LBP_descriptor_slides # LBP presentation of LBP implementation and experiments results 
```

## Dataset 

Images used for experiments are in PPM format and some examples can be found in res/images/ppm folder.


### Run experiments
To run CUDA experiments, follow the commands below:

Compile files:

`nvcc -c src/main.cu  src/PPM.cpp src/Image.cpp`

Link files:

`nvcc Image.o PPM.o  main.o -o main`

And run executable:

`./main`

## Authors
This project was carried out in collaboration with [Francesca Del Lungo](https://github.com/francidellungo) for the Parallel Computing exam.

