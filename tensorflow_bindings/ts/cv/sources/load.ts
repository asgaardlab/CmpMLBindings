// @ts-ignore
// @ts-ignore

import * as CSVParse from 'csv-parse/lib/sync';
// import {Cifar10} from 'tfjs-cifar10'
// import {Cifar10} from './loadcifar'
// import {Cifar10} from 'tfjs-cifar10-node'
import * as fs from 'fs';
import {tensor2d, Tensor, Tensor2D, tensor3d, tensor4d, Tensor4D} from "./tfimport";
import {flatten} from "lodash";
// @ts-ignore
declare module '*.png' { // @ts-ignore
    export default '' as string; }

const MNIST_DATA_PATH = "../../../data/MNIST/";
const IMDb_DATA_PATH = "../../../data/imdb/";
const CIFAR_DATA_PATH = "../../../data/cifar-10-csv/";


export const loadMNISTCSV = (file: string): void =>
    CSVParse(fs.readFileSync(file).toString())
        .slice(1)
        .map(
            (image: any): any => ({
                label: parseInt(image[0]),
                data: image.slice(1).map((num: string): number => parseInt(num))
            })
        );


export const readTxt = (file: string): number[] =>
    fs.readFileSync(file).toString().replace(/\r\n/g, '\n').split('\n').map(
        (data: any): any => parseFloat(data)
    );


const tensorizeMNISTData = (x: number[], y: number[]): [Tensor4D, Tensor2D] => {
    const x_ts: Tensor4D = tensor3d(
        flatten(x),
        [x.length, 28, 28,])
        .pad([[0, 0], [2, 2], [2, 2]])
        .div(255.0).expandDims(3);
    const y_ts = tensor2d(
        flatten(y),
        [y.length, 1]
    );
    return [x_ts, y_ts];
}


const tensorizeCIFARData = (x: number[], y: number[]): [Tensor4D, Tensor2D] => {
    const x_ts: Tensor4D = tensor4d(flatten(x), [x.length / 3, 32, 32, 3]).div(255.0);
    const y_ts = tensor2d(flatten(y), [y.length, 1]);
    return [x_ts, y_ts];
}


export const loadCIFARCSV = (file: string): number[] =>
    CSVParse(fs.readFileSync(file).toString())
        .slice(1)
        .map((image: any): any => image.map((num: string): number => parseInt(num)));


export const loadMNIST = async (): Promise<[Tensor, Tensor2D, Tensor, Tensor2D]> => {
    const store: any = {};
    store['train'] = loadMNISTCSV(MNIST_DATA_PATH + "train.csv");
    store['test'] = loadMNISTCSV(MNIST_DATA_PATH + "test.csv");

    const [x_train, y_train]: [Tensor4D, Tensor2D] = tensorizeMNISTData(
        store['train'].map((e: any) => e.data),
        store['train'].map((e: any) => e.label)
    );
    const [x_test, y_test]: [Tensor4D, Tensor2D] = tensorizeMNISTData(
        store['test'].map((e: any) => e.data),
        store['test'].map((e: any) => e.label)
    );
    return [x_train, y_train, x_test, y_test];
}


export const loadCIFAR = async (): Promise<[Tensor, Tensor2D, Tensor, Tensor2D]> => {
    const temp = loadCIFARCSV(CIFAR_DATA_PATH + "x_train_0.csv");
    var x_train_tf: Tensor4D = tensor4d(flatten(temp), [temp.length / 3, 32, 32, 3]).div(255.0);
    console.log(x_train_tf.shape);
    for (var i = 1; i < 5; i++ ) {
        const temp = loadCIFARCSV(CIFAR_DATA_PATH + `x_train_${i}.csv`);
        x_train_tf = x_train_tf.concat(tensor4d(flatten(temp), [temp.length / 3, 32, 32, 3]).div(255.0)) as Tensor4D;
        console.log(x_train_tf.shape);
    }
    const y_train = loadCIFARCSV(CIFAR_DATA_PATH + "y_train.csv");
    const y_train_tf = tensor2d(flatten(y_train), [y_train.length, 1]);

    const x_test = loadCIFARCSV(CIFAR_DATA_PATH + "x_test.csv");
    const y_test = loadCIFARCSV(CIFAR_DATA_PATH + "y_test.csv");
    const [x_test_tf, y_test_tf]: [Tensor4D, Tensor2D] = tensorizeCIFARData(x_test, y_test);

    return [x_train_tf, y_train_tf, x_test_tf, y_test_tf];
}


const tensorizeIMDbData = (x: number[], y: number[]): [Tensor2D, Tensor2D] => {
    const x_ts = tensor2d(
        flatten(x),
        [x.length, 300]
    );
    const y_ts = tensor2d(
        flatten(y),
        [y.length, 1]
    );
    return [x_ts, y_ts];
}

export const loadIMDb = async (): Promise<[Tensor, Tensor2D, Tensor, Tensor2D]> => {
    const store: any = {};
    store['train'] = loadMNISTCSV(IMDb_DATA_PATH + "train.csv");
    store['test'] = loadMNISTCSV(IMDb_DATA_PATH + "test.csv");

    const [x_train, y_train]: [Tensor2D, Tensor2D] = tensorizeIMDbData(
        store['train'].map((e: any) => e.data),
        store['train'].map((e: any) => e.label)
    );
    const [x_test, y_test]: [Tensor2D, Tensor2D] = tensorizeIMDbData(
        store['test'].map((e: any) => e.data),
        store['test'].map((e: any) => e.label)
    );
    return [x_train, y_train, x_test, y_test];
}

