# TypeScript binding for PyTorch

For running the code in GPU, you have to download the library from the official website: 
[https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip](https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip)
and add the path to `LIBTORCH` and `LD_LIBRARY_PATH`, for example: 

```shell
export LIBTORCH_GPU=PATH_TO_LIBTORCH/libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111/libtorch
export LIBTORCH=$LIBTORCH_GPU
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

Download `@arition/torch-js` and replace the `CMakeLists.txt`:

```bash
mkdir node_modules
cd node_modules
wget $(npm view @arition/torch-js@=0.12.3 dist.tarball)
tar -xvf torch-js-0.12.3.tgz
mv package/ torch-js
cd torch-js
mkdir build
cp ../../guide_to_build_deps/CMakeLists.txt ./ -f
```

Install the dependencies:

```bash
cd ../../
npm install
cd node_modules/torch-js
npm install
```

Run the code by:

```bash
npm run train gpu lenet5 200 5
npm run deploy gpu lenet5 0 5
npm run deploy cpu lenet5 0 5
```
