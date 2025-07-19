from setuptools import setup, find_packages
from typing import List

HYPPEN_E_DOT= '-e .'
def get_requirements(file_path:str)->List[str]:
    #this fn will return the list of requirements
    
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPPEN_E_DOT in requirements:
            requirements.remove(HYPPEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Atharv',
    author_email='atharvsing.edu@gmail.com',
    packages= find_packages(),
    install_requires=get_requirements('requirements.txt')
)