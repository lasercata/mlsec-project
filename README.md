# MLsec project
This project aims to re-evaluate some (CIFAR-10) models from RobustBench using different values of the epsilon radius.

## Setup
1. Create the virtual environment
```
python -m venv .venv

source .venv/bin/activate
```
(note that you will have to source it in every new terminal)

2. Install the dependencies
```
pip install -r requirements.txt
```

3. Create the `config.env`
```
cp config.example.env config.env
```

Then edit `config.env` to your needs

## Run
To run the project:
```
python main.py
```

