from setuptools import setup, find_packages

setup(
    name='dualdec',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "tqdm",
        "transformers>=4.36.0",
        "datasets>=1.6.2",
        "human_eval",
        "lade",
        "jinja2>=3.1.3"
    ],
    author='TsinghuaNLP',
    author_email='zwl23@mails.tsinghua.edu.cn, huang-yx21@mails.tsinghua.edu.cn, hanxu2022@tsinghua.edu.cn',
    description='Dual Decoding',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
