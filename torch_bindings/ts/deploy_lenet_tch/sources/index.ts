import * as CSVParse from 'csv-parse/lib/sync';
import * as fs from 'fs';
// @ts-ignore
import * as torch from "torch-js";
import {urlToHttpOptions} from "url";
import * as assert from "assert";
import {accessSync} from "fs";

const MNIST_DATA_PATH = "../../../data/MNIST/";
const CIFAR_DATA_PATH = "../../../data/cifar-10-csv/";
const CIFAR_DATA_TXT_PATH = "../../../data/cifar-10-txt/";
const IMDb_DATA_PATH = "../../../data/imdb/";
const OUT_PATH = "../../../out/pytorch";
const OUT_LR_ONLY_PATH = "../../../out/pytorch_lr_only";


export const readTxt = (file: string): number[] =>
    fs.readFileSync(file).toString().replace(/\r\n/g, '\n').split('\n').map(
        (data: any): any => parseFloat(data)
    );

export const readTxt2D = (file: string): number[][] =>
    fs.readFileSync(file).toString().replace(/\r\n/g, '\n').split('\n').map(
        (data: string): any => data.split(' ').map(
            (data: any): any => parseInt(data)
        )
    );


const average = (data: number[]): number => {
    const sum = data.reduce((a: number, b: number) => a + b, 0);
    return (sum / data.length) || 0;
}


export const lastNumber = (data: number[]): number => {
    for (var i = data.length - 1; i >= 0; i--) {
        if (!isNaN(data[i])) {
            return data[i];
        }
    }
    console.error("cannot get a valid number");
    process.exit(1);
}


const load = (file: string): void =>
    CSVParse(fs.readFileSync(file).toString())
        .slice(1)
        .map(
            (image: any): any => ({
                label: parseInt(image[0]),
                data: image.slice(1).map((num: string): number => parseInt(num) / 255.0)
            })
        );


const reshapeData = (data: any[], img_channel: number, img_row: number, img_column: number): any[] => {
    const data_reshape = [];
    var img = [];
    var c = 0;
    for (const data1d of data) {
        const one_channel_img = []
        for (var j = 0; j < data1d.length; j += img_column) {
            one_channel_img.push(data1d.slice(j, j + img_column));
        }
        console.assert(one_channel_img.length === img_row, "img's row is invalid");
        img.push(one_channel_img);
        c += 1;
        if (c === img_channel) {
            data_reshape.push(img);
            console.assert(img.length === img_channel, "img's channel is invalid");
            img = [];
            c = 0;
        }
    }
    return data_reshape;
}


const createDir = (dir: string): void => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, {recursive: true});
    }
}


const loadMNIST = (): any[] => {
    const store: any = {};
    store['test'] = load(MNIST_DATA_PATH + "/test.csv");
    store['train'] = load(MNIST_DATA_PATH + "/train.csv");
    const x_test = reshapeData(store['test'].map((e: any) => e.data), 1, 28, 28);
    const y_test = store['test'].map((e: any) => e.label);
    const x_train = reshapeData(store['train'].map((e: any) => e.data), 1, 28, 28);
    const y_train = store['train'].map((e: any) => e.label);
    return [x_test, y_test, x_train, y_train]
}


const loadCIFARCSV = (file: string, div255: boolean): number[] =>
    CSVParse(fs.readFileSync(file).toString())
        .slice(1)
        .map(
            (image: any): any =>
                image.map((num: string): number =>
                    div255 ? parseInt(num) / 255.0 : parseInt(num)
                )
        );


const loadCIFAR = (): any[] => {
    const temp = loadCIFARCSV(CIFAR_DATA_PATH + "x_train_channel_first_0.csv", true);
    var x_train = reshapeData(temp, 3, 32, 32);
    console.log(x_train.length);
    for (var i = 1; i < 5; i++ ) {
        const temp = loadCIFARCSV(CIFAR_DATA_PATH + `x_train_channel_first_${i}.csv`, true);
        x_train = x_train.concat(reshapeData(temp, 3, 32, 32));
        console.log(x_train.length);
    }
    // const y_train = loadCIFARCSV(CIFAR_DATA_PATH + "y_train.csv", false);

    var y_train = readTxt(CIFAR_DATA_TXT_PATH + "y_train.txt");
    if (isNaN(y_train[y_train.length - 1])) {
        y_train = y_train.slice(0, y_train.length - 1);
    }

    var y_test = readTxt(CIFAR_DATA_TXT_PATH + "y_test.txt");
    if (isNaN(y_test[y_train.length - 1])) {
        y_test = y_test.slice(0, y_test.length - 1);
    }

    const temp2 = loadCIFARCSV(CIFAR_DATA_PATH + "x_test_channel_first.csv", true);
    // const y_test = loadCIFARCSV(CIFAR_DATA_PATH + "y_test.csv", false);
    const x_test = reshapeData(temp2, 3,32, 32);
    return [x_test, y_test, x_train, y_train]
}


const loadIMDb = (): any[] => {
    var y_test = readTxt(IMDb_DATA_PATH + "y_test.txt");
    if (isNaN(y_test[y_test.length - 1])) {
        y_test = y_test.slice(0, y_test.length - 1);
    }
    var y_train = readTxt(IMDb_DATA_PATH + "y_train.txt");
    if (isNaN(y_train[y_train.length - 1])) {
        y_train = y_train.slice(0, y_train.length - 1);
    }
    var x_test = readTxt2D(IMDb_DATA_PATH + "x_test.txt");
    if (x_test[x_test.length - 1].length === 1 && isNaN(x_test[x_test.length - 1][0])) {
        x_test = x_test.slice(0, x_test.length - 1);
    }
    var x_train = readTxt2D(IMDb_DATA_PATH + "x_train.txt");
    if (x_train[x_train.length - 1].length === 1 && isNaN(x_train[x_train.length - 1][0])) {
        x_train = x_train.slice(0, x_train.length - 1);
    }
    return [x_test, y_test, x_train, y_train]
}


const toBatches = (data: any[], device: string, batch_size: number, int_type: boolean): any[] => {
    const batches = [];
    for (var i = 0; i < data.length; i += batch_size) {
        const temporary = data.slice(i, i + batch_size);
        if (device === "cpu") {
            if (int_type) {
                batches.push(torch.tensor(temporary, {dtype: torch.int32}).cpu());
            } else {
                batches.push(torch.tensor(temporary).cpu());
            }
        } else if (device === "gpu") {
            if (int_type) {
                batches.push(torch.tensor(temporary, {dtype: torch.int32}).cuda());
            } else {
                batches.push(torch.tensor(temporary).cuda());
            }
        } else {
            batches.push(temporary);
        }
    }
    return batches;
}


const indexOfMax = (arr: Float64Array): number => {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}


const evaluateBatches = async (model: torch.ScriptModule, x_batches: any[], y_batches: any[], device: string, int_type: boolean): Promise<number> => {
    console.assert(
        x_batches.length === y_batches.length,
        `data (${x_batches.length}) and labels (${y_batches.length}) should have the same length!`
    );
    var correct = 0;
    var total = 0;
    for (const i in x_batches) {
        // console.log(`Batch ${i}...`);
        // console.log(`${x_batches[i].length} x ${x_batches[i][0].length} x ${x_batches[i][0][0].length} x ${x_batches[i][0][0][0].length}`)
        // var x_device_cur;
        // if (device === "cpu") {
        //     if (int_type) {
        //         x_device_cur = torch.tensor(x_batches[i], {dtype: torch.int32}).cpu();
        //     } else {
        //         x_device_cur = torch.tensor(x_batches[i]).cpu();
        //     }
        // } else if (device === "gpu") {
        //     if (int_type) {
        //         x_device_cur = torch.tensor(x_batches[i], {dtype: torch.int32}).cuda();
        //     } else {
        //         x_device_cur = torch.tensor(x_batches[i]).cuda();
        //     }
        // }
        // console.log(x_device_cur);
        // console.log(x_device_cur.cpu().toObject().shape);
        // var pred = await model.forward(x_device_cur) as torch.Tensor;
        var pred = await model.forward(x_batches[i]) as torch.Tensor;
        pred = pred.cpu();
        const pred_obj = pred.toObject();
        const pred_arr = pred_obj.data as Float64Array;
        // console.log(pred_arr, pred_obj.shape);
        if (pred_obj.shape[1] === 1) {
            const pred_1d = [].concat(...pred_arr);
            // console.log(pred_1d);
            const pred_res = pred_1d.map((ele: number): number => Math.round(ele));
            // console.log(pred_res);
            // console.log(y_batches[i]);
            // console.log("==============");
            for (var j = 0; j < y_batches[i].length; j += 1) {
                if (pred_res[j] === y_batches[i][j]) {
                    correct += 1;
                }
            }
        } else {
            for (var j = 0; j < y_batches[i].length; j += 1) {
                // console.log(pred_arr.slice(j * pred_obj.shape[1], (j + 1) * pred_obj.shape[1]));
                const max_idx = indexOfMax(pred_arr.slice(j * pred_obj.shape[1], (j + 1) * pred_obj.shape[1]));
                // console.log(max_idx, y_batches[i][j], max_idx === y_batches[i][j]);
                if (max_idx === y_batches[i][j]) {
                    correct += 1;
                }
            }
        }
        total += y_batches[i].length;
        console.log(`Batch ${i}... correct/total: ${correct}/${total}`);
        // console.log("before free");
        pred.free();
        // x_device_cur.free();
    }
    console.log(correct, total);
    return correct / total;
}


const loadDataset = async (dataset: string): Promise<any[]> => {
    switch (dataset) {
        case "mnist": {
            return loadMNIST();
        }
        case "cifar": {
            return loadCIFAR();
        }
        case "imdb": {
            return loadIMDb();
        }
        default:
            console.log('dataset should be one of ["mnist", "cifar"]');
            process.exit(1);
    }
}


const loadModel = (path: string, device: string): torch.ScriptModule => {
    var model = new torch.ScriptModule(path);
    if (device === "cpu") {
        model = model.cpu();
    } else {
        model = model.cuda();
    }
    return model
}


const Model2DatasetName = new Map<string, string>([
    ["vgg16", "cifar"],
    ["lenet1", "mnist"],
    ["lenet5", "mnist"],
    ["gru", "imdb"],
    ["grurb", "imdb"],
    ["lstm", "imdb"],
]);


const isAccuracyEqual = (acc1: number, acc2: number, totalNum: number): Boolean =>
    Math.round(acc1 * totalNum) === Math.round(acc2 * totalNum);


const deploy = async (modelName: string, device: string, sgdLrOnly: boolean, trainMode: boolean, scripted: boolean): Promise<void> => {
    const dataset_name = Model2DatasetName.get(modelName);
    const end_str = trainMode ? "_train" : ""
    const prefix_str = scripted ? "scripted" : "traced"
    console.log(`${dataset_name} x ${modelName} sgdLrOnly? ${sgdLrOnly} trainMode? ${trainMode} ...`);

    const batch_size = dataset_name === "imdb" ? 256 : 128;
    const out_path = sgdLrOnly ? `${OUT_LR_ONLY_PATH}/${modelName}/ts` : `${OUT_PATH}/${modelName}/ts`;
    console.log(`out_path: ${out_path}`);
    createDir(out_path);
    const py_out_path = sgdLrOnly ? `${OUT_LR_ONLY_PATH}/${modelName}/py` : `${OUT_PATH}/${modelName}/py`;
    console.log(`py_out_path: ${py_out_path}`);

    // initialized the model first to avoid Segmentation fault when toBatches
    var model_path = `${py_out_path}/${prefix_str}_${dataset_name}-${modelName}_0${end_str}.pt`;
    if (device === "gpu" && fs.existsSync(`${py_out_path}/${prefix_str}_${dataset_name}-${modelName}_0_gpu${end_str}.pt`)) {
        model_path = `${py_out_path}/${prefix_str}_${dataset_name}-${modelName}_0_gpu${end_str}.pt`;
    }
    console.log('Loading model from: ' + model_path);
    const model = loadModel(model_path, device);
    console.log('Loaded model');

    console.log('Loading Dataset...');
    const [x_test, y_test, x_train, y_train]: any[] = await loadDataset(dataset_name);
    try {
        console.log(`x_test: ${x_test.length} x ${x_test[0].length} x ${x_test[0][0].length} x ${x_test[0][0][0].length}`)
        console.log(`x_train: ${x_train.length} x ${x_train[0].length} x ${x_train[0][0].length} x ${x_train[0][0][0].length}`)
    }
    catch (e) {
        console.log(`x_test: ${x_test.length} x ${x_test[0].length}`)
        console.log(`x_train: ${x_train.length} x ${x_train[0].length}`)
    }
    console.log(`y_test: ${y_test.length}`)
    console.log(`y_train: ${y_train.length}`)
    const testset_size = y_test.length;
    const trainset_size = y_train.length;
    console.log('Loading Dataset... done!');
    console.log('To Batches...');
    const dtype_is_int =  dataset_name === "imdb" ? true : false;

    const x_test_batches = toBatches(x_test, device, batch_size, dtype_is_int);
    const y_test_batches = toBatches(y_test, "none", batch_size, false);
    console.log("To Batches... testset done!")
    const test_acc = await evaluateBatches(model, x_test_batches, y_test_batches, device, dtype_is_int);
    console.log(`evaluateBatches for testset: ${test_acc}`);

    const x_train_batches = toBatches(x_train, device, batch_size, dtype_is_int);
    const y_train_batches = toBatches(y_train, "none", batch_size, false);
    console.log(`x_train_batches: ${x_train_batches.length}, ${x_train_batches[0].shape}`);
    console.log(`y_train_batches: ${y_train_batches.length}, ${y_train_batches[0].length}`);
    console.log("To Batches... trainset done!")
    const train_acc = await evaluateBatches(model, x_train_batches, y_train_batches, device, dtype_is_int);
    console.log(`evaluateBatches for trainset: ${train_acc}`);
    console.log('To Batches... done!');

    const train_eval_times = [];
    const test_eval_times = [];
    const testset_same_acc = [];
    const trainset_same_acc = [];
    const test_acc0_all = [];
    const train_acc0_all = [];
    const test_acc_stables = [];
    const train_acc_stables = [];
    const eval_prof_run_time = 5;
    for (var run = 0; run < 5; run++) {
        const test_error: number[] = readTxt(`${py_out_path}/testing_errors_${run}.txt`);
        const test_error_gt = lastNumber(test_error);
        const train_error: number[] = readTxt(`${py_out_path}/training_errors_${run}.txt`);
        const train_error_gt = lastNumber(train_error);

        var model_path = `${py_out_path}/${prefix_str}_${dataset_name}-${modelName}_${run}${end_str}.pt`;
        if (device === "gpu" && fs.existsSync(`${py_out_path}/${prefix_str}_${dataset_name}-${modelName}_${run}_gpu${end_str}.pt`)) {
            model_path = `${py_out_path}/${prefix_str}_${dataset_name}-${modelName}_${run}_gpu${end_str}.pt`;
        }
        console.log('Loading model from: ' + model_path);
        const model = loadModel(model_path, device);
        console.log('Loaded model');

        const train_acc0 = await evaluateBatches(model, x_train_batches, y_train_batches, device, dtype_is_int);
        const vars = await evalProf(model, x_train_batches, y_train_batches, trainset_size, device, dtype_is_int, eval_prof_run_time);
        const train_eval_time = vars[0];
        const train_accs: number[] = vars[1];
        var train_acc_stable = "true";
        for (const acc of train_accs) {
            if (!isAccuracyEqual(train_acc0, acc, trainset_size)) {
                train_acc_stable = "false";
                break;
            }
        }
        trainset_same_acc.push(`${isAccuracyEqual(train_acc0, train_error_gt, trainset_size)}`);
        train_acc0_all.push(train_acc0);
        train_eval_times.push(train_eval_time);
        train_acc_stables.push(train_acc_stable);

        const test_acc0 = await evaluateBatches(model, x_test_batches, y_test_batches, device, dtype_is_int);
        const vars2 = await evalProf(model, x_test_batches, y_test_batches, testset_size, device, dtype_is_int, eval_prof_run_time);
        const test_eval_time = vars2[0];
        const test_accs: number[] = vars2[1];
        var test_acc_stable = "true";
        for (const acc of test_accs) {
            if (!isAccuracyEqual(test_acc0, acc, testset_size)) {
                test_acc_stable = "false";
                break;
            }
        }
        testset_same_acc.push(`${isAccuracyEqual(test_acc0, test_error_gt, testset_size)}`);
        test_acc0_all.push(test_acc0);
        test_eval_times.push(test_eval_time);
        test_acc_stables.push(test_acc_stable);
    }

    fs.writeFileSync(
        `${out_path}/deploy_eval_${device}_serialization${end_str}_${prefix_str}.txt`,
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
                ]).concat(trainset_same_acc)
            .concat(
                [
                    "Test Set Accs:",
                ]).concat(test_acc0_all)
            .concat(
                [
                    "Train Set Accs:",
                ]).concat(train_acc0_all)
            .concat(
                [
                    "Test Set Accs Stable:",
                ]).concat(test_acc_stables)
            .concat(
                [
                    "Train Set Accs Stable:",
                ]).concat(train_acc_stables).join("\n")
    );
    return
}


const evalProf = async (model: torch.ScriptModule, x: any[], y: any[], datasetSize: number, device: string, dtype_is_int: boolean, runTime: number): Promise<[number, number[]]> => {
    const accs = [];
    var average_test_eval_time = 0.;
    for (var i = 0; i < runTime; i++) {
        const t0 = Date.now();
        const test_acc = await evaluateBatches(model, x, y, device, dtype_is_int);
        const t1 = Date.now();
        average_test_eval_time += (t1 - t0) / 1000.;
        accs.push(test_acc);
    }
    average_test_eval_time /= runTime;
    return [average_test_eval_time, accs];
}


const main = async (): Promise<void> => {
    const model_name = process.argv[3].toLowerCase();
    assert.ok(['vgg16', 'resnet20', 'lenet1', 'lenet5', 'gru', 'grurb', 'lstm'].includes(model_name));
    var sgd_lr_only = false;
    if (process.argv.length === 5 && process.argv[4] === "sgd_lr_only") {
        sgd_lr_only = true;
    }
    var device;
    switch (process.argv[2]) {
        case 'cpu':
            device = "cpu";
            break;
        case "gpu":
            device = "gpu";
            break;
        default:
            throw new Error(
                'Use "cpu"/"gpu" to deploy the model in cpu/cuda.'
            );
    }
    // await deploy(model_name, device, sgd_lr_only, true, true);
    await deploy(model_name, device, sgd_lr_only, false, true);
    // await deploy(model_name, device, sgd_lr_only, true, false);
    // await deploy(model_name, device, sgd_lr_only, false, false);

    return
};

main().catch(
    (e: any): void => {
        console.log(e)
        process.exit(1);
    }
);

// main();
