import {tf, Tensor, layers, sequential, Sequential} from './tfimport';


const VGG16_CONFIG = [64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0];


export const buildVGG16 = (x_train: Tensor, sgdLrOnly: boolean): Sequential => {
    const model = sequential();
    var first = true;
    const c_in = Array.from(x_train.shape.slice(1, 4));
    console.log(c_in);

    for (const config of VGG16_CONFIG) {
        console.log(config)
        if (config === 0) {
            model.add(layers.maxPooling2d({poolSize: [2, 2]}))
        } else {
            if (first) {
                model.add(
                    layers.conv2d({
                        filters: config,
                        kernelSize: 3,
                        padding: 'same',
                        inputShape: c_in,
                        useBias: false,
                    })
                );
                first = false;
            } else {
                model.add(
                    layers.conv2d({
                        filters: config,
                        kernelSize: 3,
                        padding: 'same',
                        useBias: false,
                    })
                );
            }
            model.add(layers.batchNormalization())
            model.add(layers.reLU())
        }
    }

    model.add(layers.flatten())
    model.add(layers.dense({units: 4096, activation: 'relu'}))
    model.add(layers.dropout({rate: 0.5}))
    model.add(layers.dense({units: 4096, activation: 'relu'}))
    model.add(layers.dropout({rate: 0.5}))

    var optimizer;
    if (sgdLrOnly) {
        model.add(
            layers.dense({
                units: 10,
                activation: 'softmax',
            })
        );
        optimizer = tf.train.sgd(5e-2);
    } else {
        model.add(
            layers.dense({
                units: 10,
                activation: 'softmax',
                kernelRegularizer: tf.regularizers.l2({l2: 1e-4}),  // weight decay
                biasRegularizer: tf.regularizers.l2({l2: 1e-4}),  // weight decay
            })
        );
        optimizer = tf.train.momentum(1e-1, 9e-1);
    }
    model.summary();
    model.compile({
        optimizer: optimizer,
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    })
    return model;
};
