# COMP4332 Project 1 (SpaCy + LSTM)
This repository includes a pipeline using SpaCy module for string embedding and train models based on different shapes of embedded features. 
## Quickstart
Create an new conda virtual environment 
```
conda create -n fasternet python=3.9.12 -y
conda activate fasternet
```
Clone this repo and install the required packages
```
git clone https://github.com/DanielSHKao/comp4332_project1.git
pip install -r requirements.txt
pip install -U scikit-learn
```
Train model with command
```
python main.py
```
Visit proj1.yaml for customized model and training strategy.
To adjust the batch size or gpu index, please refer to the following command
```
python main.py -b 32 -g 0,1
```
We provide four types of embedding methods, including 'word', 'sentence', 'subtext', and 'subsentence'. 
Available models can be found in `models/registry.py`.