import * as tf from '@tensorflow/tfjs-node';
import {
    tensor2d, Tensor, Tensor2D, Tensor3D, Tensor4D, tensor3d, tensor4d, CustomCallback, LayersModel, loadLayersModel,
    layers, sequential, Sequential
} from "@tensorflow/tfjs-node";

import {SaveResult} from '@tensorflow/tfjs-core/dist/io/io';
const DEVICE = "cpu";

export {
    tf, tensor2d, Tensor, Tensor2D, Tensor3D, Tensor4D, tensor3d, tensor4d, CustomCallback, LayersModel, loadLayersModel,
    layers, sequential, Sequential, SaveResult, DEVICE
};
