from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirments(file_path:str)->List[str]:
    '''
    This function will retun the list of requirments
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements    


setup(
    name= 'face impression recognition classification',
    version = '0.0.1',
    author = 'Meghdad',
    author_email = 'mehdizadeh.iust@gmail.com',
    url="https://github.com/Meghdad-DTU/Facial_Impression_Recognition_Calassification",
    packeges = find_packages(where='src'),
    install_requires = get_requirments('requirements.txt')
    )