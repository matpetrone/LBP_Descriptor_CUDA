# LBP Descriptor CUDA
Project for the Parallel Computing exam at the University of Florence.

The sequential and first parallel implementations for the same project can be found here: [LBP descriptor](https://github.com/matpetrone/LBP_Descriptor).

Local binary patterns (LBP) is a type of visual descriptor used for classification in computer vision. It has since been found to be a powerful feature for texture classification.

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

To generate matrix from images:

` python src/imageToCsv.py -f <image_filename> [-id <images_directory> -od <csv_matrix_output_directory>]`

## Authors
This project was carried out in collaboration with [Francesca Del Lungo](https://github.com/francidellungo) for the Parallel Computing exam.

## WORK IN PROGRESS...
