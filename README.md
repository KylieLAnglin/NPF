# template: Template for new projects

## What is it?
______
This is the set-up I have found helpful for avoid struggling package imports for testing. The approach is adapted from [this](https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944) Stackoverflow answer, "Tired of sys.path hacks?". 

## File structure
____

```
template
|
data
    |_ raw
    |_ clean
|
setup.py
|
template
    |
    --library
        |_ __init__.py
        |_ example.py
    |
    --README.md
    |
    --tests
        |_ __init__.py
        |_ test_example.py

Where template represents the name of my project. I may also have additional folders containing code within the second template folder. Each of these will also have an __init__.py. 

```
## Set up
___
1. Create file structure and setup.py
2. Create new virtual environment for project (I use pyenv)
3. In root template folder, run 
```
pip install -e .
```
4. Where necessary, add template. to beginning of import statements. 

For example, to test example.py in test_example.py,
```
from template.library import example
```
## Git
In practice, I place my git repo in the second template folder, meaning I don't track my data with git. 


