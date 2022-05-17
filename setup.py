import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='classifier_divergence',
    version='0.0.1',
    author='Anthony Sicilia',
    author_email='anthonysicilia.contact@gmail.com',
    description='Compute model-dependent distribution divergence.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/anthonysicilia/classifier-divergence',
    project_urls = {
        "Bug Tracker": "https://github.com/anthonysicilia/classifier-divergence/issues"
    },
    license='MIT',
    packages=['topic_modeling'],
    install_requires=['tqdm', 'torch', 'torchvision'],
)
