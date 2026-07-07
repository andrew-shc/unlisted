git push git@github.com:andrew-shc/unlisted.git main:metrology_ir


after installing n-pbir
```python
pip uninstall psdr-jit
cd ../psdr-jit
git submodule update --init --recursive
pip install --no-build-isolation -ve . --config-settings=cmake.args="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# verify
python -c "import psdr_jit; print(psdr_jit.__file__)"
```
