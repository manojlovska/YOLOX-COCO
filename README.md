## Steps to reproduce
Step 1: Clone the main repository and the submodule
'''shell
    git pull --recurse submodules
    '''
Step 2: Create and activate virtual environment with conda
    '''shell
        conda create -n env python=3.8.5
        conda activate env
        '''
Step 3: Install the requriements
    '''shell
        pip install -r requirements.txt
        pip install -ve YOLOX
        '''
Step 4: Pull the dataset
    '''shell
        dvc pull
        '''
Step 5: Train the model
    '''shell
        ./train.sh
        '''
