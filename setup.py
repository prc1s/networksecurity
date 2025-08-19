from setuptools import setup,find_packages
from typing import List


def get_requirements() -> List[str]:
    requirement_lst:List[str] = []
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found")
    
    except Exception as e:
        raise e 
    
    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Mohammed Aljowaie",
    author_email="mohammed.bander@outlook.com",
    packages=find_packages(),
    install_requires=get_requirements()
)