# Source Code	

`main.py` serves as the entrance of our framework, and there are three main packages. 

### Structure

- `helpers\`
  - `BaseReader.py`: read dataset csv into DataFrame and append necessary information (e.g. interaction history)
  - `BaseRunner.py`: control the training and evaluation process of a model
  - `...`: customize helpers with specific functions
- `models\`
  - `BaseModel.py`: basic model classes and dataset classes, with some common functions of a model
  - `...`: customize models inherited from classes in *BaseModel*
- `utils\`
  - `layers.py`: common modules for model definition (e.g. attention)
  - `utils.py`: some utils functions
- `main.py`: main entrance, connect all the modules
- `exp.py`: repeat experiments in *run.sh* and save averaged results to csv 