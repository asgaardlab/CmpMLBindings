import {tf, Tensor, layers, sequential, Sequential} from './tfimport';
import {constant} from "lodash";


export const buildLeNet5 = (x_train: Tensor, sgdLrOnly: boolean): Sequential => {
    const model = sequential();

    // First convolutional layer, receive 28x28x1 inputs, applies 6 filters
    const c_in = Array.from(x_train.shape.slice(1, 4));
    console.log(c_in);
    model.add(
        layers.conv2d({
            // inputShape: [32, 32, 1],
            inputShape: c_in,
            filters: 6,
            kernelSize: 5,
            strides: 1,
            activation: 'tanh',
        })
    );

    model.add(layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Second convolutional layer, applies 16 layers
    model.add(
        layers.conv2d({
            filters: 16,
            kernelSize: 5,
            strides: 1,
            activation: 'tanh',
        })
    );

    model.add(layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    model.add(
        layers.conv2d({
            filters: 120,
            kernelSize: 5,
            strides: 1,
            activation: 'tanh',
        })
    );

    // Flattens from 2d to 1d
    model.add(layers.flatten());

    // Fully connected layer
    model.add(
        layers.dense({
            units: 84,
            activation: 'tanh'
        })
    );

    // Fully connected layer
    model.add(
        layers.dense({
            units: 10,
            activation: 'softmax'
        })
    );

    model.summary();
    // Compiling the model with rmsprop optimizer
    var optimizer;
    if (sgdLrOnly) {
        optimizer = tf.train.sgd(0.05);
    } else {
        optimizer = tf.train.momentum(0.05, 0.9);
    }
    model.compile({
        optimizer: optimizer,
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    })
    return model;
};


export const buildLeNet1 = (x_train: Tensor, sgdLrOnly: boolean): Sequential => {
    const model = sequential();

    // First convolutional layer, receive 28x28x1 inputs, applies 4 filters
    const c_in = Array.from(x_train.shape.slice(1, 4));
    console.log(c_in);
    model.add(
        layers.conv2d({
            // inputShape: [32, 32, 1],
            inputShape: c_in,
            filters: 4,
            kernelSize: 5,
            strides: 1,
            activation: 'tanh',
        })
    );

    model.add(layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Second convolutional layer, applies 12 layers
    model.add(
        layers.conv2d({
            filters: 12,
            kernelSize: 5,
            strides: 1,
            activation: 'tanh',
        })
    );

    model.add(layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Flattens from 2d to 1d
    model.add(layers.flatten());

    // Fully connected layer
    model.add(
        layers.dense({
            units: 10,
            activation: 'softmax'
        })
    );

    model.summary();
    // Compiling the model with rmsprop optimizer
    var optimizer;
    if (sgdLrOnly) {
        optimizer = tf.train.sgd(0.05);
    } else {
        optimizer = tf.train.momentum(0.05, 0.9);
    }
    model.compile({
        optimizer: optimizer,
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    })
    return model;
};
