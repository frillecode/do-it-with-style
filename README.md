<br />

  <h1 align="center">Do it with style</h1> 

  <h2 align="center">Embedding classical European paintings using Neural Style Transfer </h2> 
  <p align="center">
    Jan Kostkan & Frida Hæstrup 


This repository contains the code related to our exam paper in Data Science at Aarhus University. The project uses Neural Style Transfer to create and analyse object-agnostic representations of digitized images of paintings. 


## Structure
This repository has the following directory structure:

```bash
do-it-with-style/  
├── src/
│   └── StyleExtractor.py # NST: extracting style images
│   └── sampling.py # sampling data
│   └── extract_embeddings.py # extracting image embeddings
│   └── embedding_cluster.py # computing clusters
│   └── cross-validation.py # CNN classification model
├── analysis/  
│   └── cross-validation_results.py # summarizing classification results
│   └── prototypical_paintings.py # extracting central images
```

## Technicalities
To run scripts within this repository, we recommend cloning the repository and installing relevant dependencies in a virtual ennvironment:
```bash
$ git clone https://github.com/frillecode/do-it-with-style
$ cd do-it-with-style
$ bash ./create_venv.sh 
```

To perform Neural Style Transfer and extract style images, run the following from the command-line:
```bash
$ cd src
$ python3 StyleExtractor.py -ip "path/to/image_folder"
```


