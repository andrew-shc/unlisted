

For neural pbir
```sh

# pip install --upgrade setuptools && pip install -r requirements.txt --no-build-isolation

# pip install --force-reinstall setuptools

# pip install --no-build-isolation "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# git clone --recursive https://github.com/NVlabs/tiny-cuda-nn/ /tmp/tinycudann
# cd /tmp/tinycudann/bindings/torch
# sed -i 's/from pkg_resources import parse_version/from packaging.version import Version as parse_version/' /tmp/tinycudann/bindings/torch/setup.py
# python setup.py install

# conda install setuptools
# conda install -c conda-forge setuptools


cd DigitalTwinCatalog/neural_pbir
pip install "setuptools<72"

pip install -r requirements.txt --no-build-isolation

pip install scikit-build-core
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install psdr-jit==0.2.1 --no-build-isolation
pip install -r requirements.txt --no-build-isolation

```
