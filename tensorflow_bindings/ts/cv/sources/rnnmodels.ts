import {tf, Tensor, layers, sequential, Sequential} from './tfimport';


const NUM_WORDS = 10000;
const EMBEDDING_VEC_LEN = 300;


export const buildGRURB = (x_train: Tensor): Sequential => {
    const model = sequential();
    console.log(x_train.shape[1]);
    model.add(layers.embedding({inputDim: NUM_WORDS, outputDim: EMBEDDING_VEC_LEN, inputLength: x_train.shape[1]}));
    // model.add(layers.dropout({rate: 0.5}));
    model.add(layers.gru({units: 512}));
    model.add(layers.dropout({rate: 0.5}));
    model.add(layers.dense({units: 1, activation: 'sigmoid'}))

    model.summary();
    const optimizer = tf.train.adam(3e-4);
    model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })
    return model;
};


export const buildLSTM = (x_train: Tensor): Sequential => {
    const model = sequential();
    console.log(x_train.shape[1]);
    model.add(layers.embedding({inputDim: NUM_WORDS, outputDim: EMBEDDING_VEC_LEN, inputLength: x_train.shape[1]}));
    // model.add(layers.dropout({rate: 0.5}));
    model.add(layers.lstm({units: 512}));
    model.add(layers.dropout({rate: 0.5}));
    model.add(layers.dense({units: 1, activation: 'sigmoid'}))

    model.summary();
    const optimizer = tf.train.adam(8e-5);
    model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })
    return model;
};


