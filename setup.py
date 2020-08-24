import os.path
import sys
import xml.etree.ElementTree
from pathlib import Path

import xacro
from setuptools import setup, find_packages

ENTRY_POINTS = [('main', 'main', 'main')]
SHARE_DIRS = [('launch', '*.launch.py'), ('config', '*.rviz'), ('worlds', '*.world'), ('models', '*')]

ROOT = xml.etree.ElementTree.parse('package.xml').getroot()
PACKAGE_NAME = ROOT.findall('name')[-1].text

ALL_MAINTAINERS = ROOT.findall('maintainer')
MAINTAINERS = [maintainer.text for maintainer in ALL_MAINTAINERS]
MAINTAINER_EMAILS = [maintainer.attrib['email'] for maintainer in ALL_MAINTAINERS]

AUTHORS = ROOT.findall('author')
AUTHOR_NAMES = [author.text for author in AUTHORS]
AUTHOR_EMAILS = [author.attrib['email'] for author in AUTHORS]

DATA_FILES = [(f'share/{PACKAGE_NAME}', ['package.xml']),
              ('share/ament_index/resource_index/packages/', [f'resources/{PACKAGE_NAME}'])]
DATA_FILES += [(os.path.join('share', PACKAGE_NAME, str(directory)),
                [str(file) for file in directory.rglob(pattern) if not file.is_dir() and file.parent == directory])
               for folder, pattern in SHARE_DIRS for directory in Path(folder).rglob('**')]

BUILD_DIR = next((sys.argv[i + 1] for i, arg in enumerate(sys.argv) if arg == '--build-directory'), None)
if BUILD_DIR:
    os.makedirs(BUILD_DIR, exist_ok=True)
    for path, files in DATA_FILES:
        for file in files:
            if file.endswith('.xacro'):
                new_file = os.path.join(BUILD_DIR, file.replace('.xacro', ''))
                with xacro.open_output(new_file) as fd:
                    fd.write(xacro.process(file))
                files.remove(file)
                files.append(new_file)

INSTALL_REQUIRES = ['setuptools']
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as file:
        INSTALL_REQUIRES += [line.strip() for line in file.readlines()]

TESTS_REQUIRE = ['pytest']
if os.path.isfile('test-requirements.txt'):
    with open('test-requirements.txt') as file:
        TESTS_REQUIRE += [line.strip() for line in file.readlines()]

setup(
    name=PACKAGE_NAME,
    version=ROOT.findall('version')[-1].text,
    packages=find_packages(),
    data_files=DATA_FILES,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    zip_safe=True,
    author=', '.join(AUTHOR_NAMES),
    author_email=', '.join(AUTHOR_EMAILS),
    maintainer=', '.join(MAINTAINERS),
    maintainer_email=', '.join(MAINTAINER_EMAILS),
    keywords=['ROS'],
    description=ROOT.findall('description')[-1].text,
    license=ROOT.findall('license')[-1].text,
    entry_points={'console_scripts': [f'{cmd} = {PACKAGE_NAME}.{file}:{func}' for cmd, file, func in ENTRY_POINTS]}
)
