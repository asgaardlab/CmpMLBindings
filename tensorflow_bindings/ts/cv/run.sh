conda activate tensorflow_gpu
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
rm -rf build/ node_modules/
npm install

# for LeNet1
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train lenet1 200 5
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet1
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet1
# sgd with learning rate only
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train lenet1 200 5 sgd_lr_only
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet1 0 0 sgd_lr_only
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet1 0 0 sgd_lr_only

# for LeNet5
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train lenet5 200 5
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet5
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet5
# sgd with learning rate only
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train lenet5 200 5 sgd_lr_only
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet5 0 0 sgd_lr_only
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lenet5 0 0 sgd_lr_only

# for VGG16
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train vgg16 200 5
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy vgg16
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy vgg16
# sgd with learning rate only
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train vgg16 200 5 sgd_lr_only
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy vgg16 0 0 sgd_lr_only
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy vgg16 0 0 sgd_lr_only

# for LSTM
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train lstm 30 5
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lstm
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy lstm

# for GRU (not supported)
#cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
#npm run clean
#npm run build
#npm run train gru 30 5
#cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
#npm run clean
#npm run build
#npm run deploy gru
#cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
#npm run clean
#npm run build
#npm run deploy gru

# for GRU (reset_after=false)
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run train grurb 30 5
cp -f ./sources/tfgpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy grurb
cp -f ./sources/tfcpu.ts ./sources/tfimport.ts
npm run clean
npm run build
npm run deploy grurb
