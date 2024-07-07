// Random: https://github.com/tensorflow/tfjs/issues/2219
import {loadCIFAR, loadIMDb, loadMNIST, readTxt} from './load';
import {buildLeNet5, buildLeNet1} from './lenetmodel';
import {buildVGG16} from './vggmodel';
import {saveModel} from './save';
import {importModel} from './import';
import {
    tf,
    Tensor,
    tensor2d,
    Tensor2D,
    Tensor3D,
    Tensor4D,
    tensor3d,
    CustomCallback,
    Sequential,
    DEVICE,
    LayersModel,
    layers
} from "./tfimport";
import * as fs from 'fs';
import * as assert from "assert";
import {L1L2, L1L2Args, Regularizer} from "@tensorflow/tfjs-layers/dist/regularizers";
import {buildGRURB, buildLSTM} from "./rnnmodels";
import * as seedrandom from 'seedrandom';


const OUT_PATH = "../../../out/tensorflow";
const OUT_LR_ONLY_PATH = "../../../out/tensorflow_lr_only";
const SEEDS_PATH = "../../../random_seeds.txt";


const average = (data: number[]): number => {
    const sum = data.reduce((a: number, b: number) => a + b, 0);
    return (sum / data.length) || 0;
}


const isAccuracyEqual = (acc1: number, acc2: number, totalNum: number): Boolean =>
    Math.round(acc1 * totalNum) === Math.round(acc2 * totalNum);


export const lastNumber = (data: number[]): number => {
    for (var i = data.length - 1; i >= 0; i--) {
        if (!isNaN(data[i])) {
            return data[i];
        }
    }
    console.error("cannot get a valid number");
    process.exit(1);
}


const createDir = (dir: string): void => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, {recursive: true});
    }
}


const setRandomSeeds = (seed: number): void => {
    seedrandom(seed.toString());
}


const loadDataset = async (dataset: string): Promise<[Tensor, Tensor2D, Tensor, Tensor2D]> => {
    switch (dataset) {
        case "mnist": {
            return await loadMNIST();
        }
        case "cifar": {
            return await loadCIFAR();
        }
        case "imdb": {
            return await loadIMDb();
        }
        default:
            console.log('dataset should be one of ["mnist", "cifar", "imdb"]');
            process.exit(1);
    }
}


const createModel = async (modelName: string, x_train: Tensor, sgdLrOnly: boolean): Promise<Sequential> => {
    switch (modelName) {
        case "lenet1": {
            return buildLeNet1(x_train, sgdLrOnly);
        }
        case "lenet5": {
            return buildLeNet5(x_train, sgdLrOnly);
        }
        case "vgg16": {
            return buildVGG16(x_train, sgdLrOnly);
        }
        case "lstm": {
            return buildLSTM(x_train);
        }
        case "resnet20": {
            console.log('resnet20 is not implemented');
            process.exit(1);
        }
        case "gru": {
            console.log('gru with reset_after=true is not supported');
            process.exit(1);
        }
        case "grurb": {
            return buildGRURB(x_train);
        }
        default:
            console.log('modelName should be one of ["lenet1", "lenet5", "vgg16", "lstm", "gru", "grurb"]');
            process.exit(1);
    }
}


const Model2DatasetName = new Map<string, string>([
    ["vgg16", "cifar"],
    ["lenet1", "mnist"],
    ["lenet5", "mnist"],
    ["gru", "imdb"],
    ["grurb", "imdb"],
    ["lstm", "imdb"],
]);


const DatasetBatchSize = new Map<string, number>([
    ["cifar", 128],
    ["mnist", 128],
    ["imdb", 256],
]);


const LossFuncMap = new Map<string, string>([
    ["cifar", 'sparseCategoricalCrossentropy'],
    ["mnist", 'sparseCategoricalCrossentropy'],
    ["imdb", 'binaryCrossentropy'],
]);


const train = async (modelName: string, epochs: number, runNum: number, sgdLrOnly: boolean, startFrom: number): Promise<void> => {
    const dataset_name = Model2DatasetName.get(modelName);
    const batch_size = DatasetBatchSize.get(dataset_name);
    const seeds: number[] = readTxt(SEEDS_PATH);
    console.log(`${dataset_name} x ${modelName} epochs: ${epochs}, runNum: ${runNum}..., batch_size: ${batch_size}, sgdLrOnly? ${sgdLrOnly}`);
    const [x_train, y_train, x_test, y_test]: [Tensor, Tensor2D, Tensor, Tensor2D] = await loadDataset(dataset_name);
    console.log("training set shapes:", x_train.shape, y_train.shape);
    console.log("testing set shapes:", x_test.shape, y_test.shape);

    const out_path = sgdLrOnly ? `${OUT_LR_ONLY_PATH}/${modelName}/ts` : `${OUT_PATH}/${modelName}/ts`;
    console.log(`out_path: ${out_path}`);
    createDir(out_path);

    for (var i = startFrom; i < runNum; i++) {
        const model = await createModel(modelName, x_train, sgdLrOnly);
        const train_accs: number[] = [];
        const test_accs: number[] = [];
        const seed = seeds[i];
        setRandomSeeds(seed);
        console.log(`seed: ${seed}`);

        const t0 = Date.now();
        var eval_time = 0.0;
        const history = await model.fit(x_train, y_train, {
            batchSize: batch_size, epochs: epochs, verbose: 0, callbacks: [
                new CustomCallback({
                    onEpochEnd: async (epoch: number): Promise<void> => {
                        const temp = Date.now();
                        const train_res = model.evaluate(x_train, y_train, {"verbose": 0}) as tf.Scalar[];
                        const test_res = model.evaluate(x_test, y_test, {"verbose": 0}) as tf.Scalar[];
                        // tslint:disable-next-line:typedef
                        const [train_loss, train_acc] = train_res.map((s: tf.Scalar) => s.dataSync()[0]);
                        // tslint:disable-next-line:typedef
                        const [test_loss, test_acc] = test_res.map((s: tf.Scalar) => s.dataSync()[0]);
                        console.log(`${epoch}: train_acc - ${train_acc}, test_acc ${test_acc}`);
                        train_accs.push(train_acc);
                        test_accs.push(test_acc);
                        eval_time += (Date.now() - temp) / 1000.;
                    },
                })
            ],
        });
        const t1 = Date.now();
        const total_time = (t1 - t0) / 1000.;
        const training_time = total_time - eval_time;
        const train_res = model.evaluate(x_train, y_train, {"verbose": 0}) as tf.Scalar[];
        const test_res = model.evaluate(x_test, y_test, {"verbose": 0}) as tf.Scalar[];

        console.log(
            `\nEvaluation result for training set:\n` +
            `  Loss = ${train_res[0].dataSync()[0].toFixed(16)}; ` +
            `Accuracy = ${train_res[1].dataSync()[0].toFixed(16)}`);
        console.log(
            `\nEvaluation result for test set:\n` +
            `  Loss = ${test_res[0].dataSync()[0].toFixed(16)}; ` +
            `Accuracy = ${test_res[1].dataSync()[0].toFixed(16)}`);
        // tslint:disable-next-line:typedef
        const [train_loss, train_acc] = train_res.map((s: tf.Scalar) => s.dataSync()[0]);
        // tslint:disable-next-line:typedef
        const [test_loss, test_acc] = test_res.map((s: tf.Scalar) => s.dataSync()[0]);

        fs.writeFileSync(
            `${out_path}/time_cost_${i}.txt`,
            [
                "Training Time:", training_time,
                "Total Time:", total_time,
                "Eval Time:", eval_time,
                "Testing Loss:", test_loss,
                "Testing Acc:", test_acc,
                "Training Loss:", train_loss,
                "Training Acc:", train_acc,
                "Seed:", seed,
            ].join("\n")
        );

        fs.writeFileSync(`${out_path}/training_errors_${i}.txt`, train_accs.join("\n"));
        fs.writeFileSync(`${out_path}/testing_errors_${i}.txt`, test_accs.join("\n"));
        await saveModel(model, `file://${out_path}/model_${i}`);
        model.dispose();
    }
};


const evalProf = async (model: LayersModel, x: Tensor, y: Tensor2D): Promise<[number, number[]]> => {
    const accs = [];
    var average_test_eval_time = 0.;

    for (var i = 0; i < 5; i++) {
        const t0 = Date.now();
        const res = model.evaluate(x, y, {"verbose": 0}) as tf.Scalar[];
        const t1 = Date.now();
        const [loss, acc]: number[] = res.map((s: tf.Scalar) => s.dataSync()[0]);
        average_test_eval_time += (t1 - t0) / 1000.;
        accs.push(acc);
        assert.ok(isAccuracyEqual(acc, accs[0], y.shape[0]), `acc ${acc} != accs[0]${accs[0]}`);
    }
    average_test_eval_time /= 5;
    return [average_test_eval_time, accs];
}


const deploy = async (modelName: string, sgdLrOnly: boolean): Promise<void> => {

    class L2  {

        static className: string = 'L2';

        constructor(config: L1L2Args) {
            // super(config)
            return tf.regularizers.l1l2(config)
        }
    }
    // @ts-ignore
    tf.serialization.registerClass(L2);

    const dataset_name = Model2DatasetName.get(modelName);
    console.log(`${dataset_name} x ${modelName} in ${DEVICE}...`);
    const [x_train, y_train, x_test, y_test]: [Tensor, Tensor2D, Tensor, Tensor2D] = await loadDataset(dataset_name);


    const out_path = sgdLrOnly ? `${OUT_LR_ONLY_PATH}/${modelName}/ts` : `${OUT_PATH}/${modelName}/ts`;
    console.log(`out_path: ${out_path}`);
    createDir(out_path);
    const py_out_path = sgdLrOnly ? `${OUT_LR_ONLY_PATH}/${modelName}/py` : `${OUT_PATH}/${modelName}/py`;
    console.log(`py_out_path: ${py_out_path}`);

    const train_eval_times = [];
    const test_eval_times = [];
    const testset_same_acc = [];
    const trainset_same_acc = [];

    for (var run = 0; run < 5; run++) {
        const test_error: number[] = readTxt(`${py_out_path}/testing_errors_${run}.txt`);
        const test_error_gt = lastNumber(test_error);
        const train_error: number[] = readTxt(`${py_out_path}/training_errors_${run}.txt`);
        const train_error_gt = lastNumber(train_error);

        const model_path = `${py_out_path}/${dataset_name}-${modelName}_${run}_tfjs/model.json`;
        console.log('Loading model from: ' + model_path);
        const model = await importModel(model_path);
        model.compile({
            optimizer: tf.train.momentum(0.05, 0.9),
            loss: LossFuncMap.get(dataset_name),
            metrics: ['accuracy']
        });
        console.log('Loaded model');

        const train_res0 = model.evaluate(x_train, y_train, {"verbose": 0}) as tf.Scalar[];
        const [train_loss0, train_acc0]: number[] = train_res0.map((s: tf.Scalar) => s.dataSync()[0]);
        const vars = await evalProf(model, x_train, y_train);
        const train_eval_time = vars[0];
        const train_accs: number[] = vars[1];
        for (const acc of train_accs) {
            assert.ok(isAccuracyEqual(train_acc0, acc, y_train.shape[0]), `train_acc0 ${train_acc0} != acc${acc}`);
        }
        trainset_same_acc.push(`${isAccuracyEqual(train_acc0, train_error_gt, y_train.shape[0])}`);
        train_eval_times.push(train_eval_time);

        const test_res0 = model.evaluate(x_test, y_test, {"verbose": 0}) as tf.Scalar[];
        const [test_loss0, test_acc0]: number[] = test_res0.map((s: tf.Scalar) => s.dataSync()[0]);
        const vars2 = await evalProf(model, x_test, y_test);
        const test_eval_time = vars2[0];
        const test_accs: number[] = vars2[1];
        for (const acc of test_accs) {
            assert.ok(isAccuracyEqual(test_acc0, acc, y_test.shape[0]), `test_acc0 ${test_acc0} != acc${acc}`);
        }
        testset_same_acc.push(`${isAccuracyEqual(test_acc0, test_error_gt, y_test.shape[0])}`);
        test_eval_times.push(test_eval_time);
    }

    fs.writeFileSync(
        `${out_path}/deploy_eval_${DEVICE}_serialization.txt`,
        [
            "Test average:", average(test_eval_times),
            "Test:",
        ].concat(test_eval_times)
            .concat(
                [
                    "Train average:", average(train_eval_times),
                    "Train:",
                ]).concat(train_eval_times)
            .concat(
                [
                    "Test Set Same Acc:",
                ]).concat(testset_same_acc)
            .concat(
                [
                    "Train Set Same Acc:",
                ]).concat(trainset_same_acc).join("\n")
    );
};


const main = async (): Promise<void> => {
    const x = layers.conv2d({
        inputShape: [32, 32, 1],
        filters: 4,
        kernelSize: 5,
        strides: 1,
        activation: 'tanh',
        kernelInitializer: "glorotNormal",
    });
    console.log(x.getConfig());
    const model = process.argv[3].toLowerCase();
    assert.ok(['vgg16', 'resnet20', 'lenet1', 'lenet5', 'gru', 'grurb', 'lstm'].includes(model));
    var sgd_lr_only = false;
    if (process.argv.length === 7 && process.argv[6] === "sgd_lr_only") {
        sgd_lr_only = true;
    }

    var start_from = 0;
    if (process.argv.length === 8) {
        start_from = parseInt(process.argv[7]);
    }
    console.log("start_from:", start_from);

    switch (process.argv[2]) {
        case 'train':
            return train(model, parseInt(process.argv[4]), parseInt(process.argv[5]), sgd_lr_only, start_from);
        case 'deploy':
            return deploy(model, sgd_lr_only);
        default:
            throw new Error(
                'You can train or deploy models. model should be ' +
                `one of ["lenet1", "lenet5", "vgg16", "lstm", "gru", "grurb"]` +
                '- train: ' +
                '\tnpm run train model number_of_epochs number_of_run' +
                '- performance of deploying' +
                '\tnpm run deploy model'
            );
    }
};


main().catch(
    (e: any): void => {
        console.error(e);
        process.exit(1);
    }
);
