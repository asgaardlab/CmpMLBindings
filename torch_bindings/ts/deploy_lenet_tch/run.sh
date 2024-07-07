#conda install -c conda-forge cudnn==8.1.0.77
#conda install cudatoolkit-dev==11.2.2
export LIBTORCH=$LIBTORCH_GPU
export PATH="${LIBTORCH}:${CONDA_ENV_PYTORCH_LIB}:${PATH}"
export LD_LIBRARY_PATH="${LIBTORCH}:${CONDA_ENV_PYTORCH_LIB}:${LD_LIBRARY_PATH}"

mkdir node_modules
cd node_modules
wget $(npm view @arition/torch-js@=0.12.3 dist.tarball)
tar -xvf torch-js-0.12.3.tgz
mv package/ torch-js
cd torch-js
mkdir build
cp ../../guide_to_build_deps/CMakeLists.txt ./ -f
#cp -r ${LIBTORCH} build/
cd ../../
npm install
cd node_modules/torch-js
npm install
# ignore the error of "husky install"

cd ../../
npm run build

#npm run gpu lenet1
#npm run cpu lenet1
# sgd with learning rate only
npm run gpu lenet1 sgd_lr_only
npm run cpu lenet1 sgd_lr_only

#npm run gpu lenet5
#npm run cpu lenet5
# sgd with learning rate only
npm run gpu lenet5 sgd_lr_only
npm run cpu lenet5 sgd_lr_only

#npm run gpu vgg16
#npm run cpu vgg16
# sgd with learning rate only
npm run gpu vgg16 sgd_lr_only
npm run cpu vgg16 sgd_lr_only

npm run gpu lstm
npm run cpu lstm

npm run gpu gru
npm run cpu gru
