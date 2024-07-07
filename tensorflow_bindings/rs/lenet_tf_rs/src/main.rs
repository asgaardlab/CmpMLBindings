// mod momentum_optimizer;
use structopt::StructOpt;
use std::fs::{File, create_dir_all};
use anyhow::Result;
use std::time::{Instant};
use std::io::{BufWriter, Write, BufReader, prelude::*};
use std::str::FromStr;
use crate::serialization_model::SerializedModel;
use tensorflow as tf;

mod dataset_loader;
mod mnist_model;
// mod vgg_model;
// mod graph_model;
mod serialization_model;


static OUT_PATH: &str = "../../../out/tensorflow";
static OUT_LR_ONLY_PATH: &str = "../../../out/tensorflow_lr_only";


#[cfg(feature = "gpu")]
fn get_device() -> &'static str {
    return "gpu";
}


#[cfg(not(feature = "gpu"))]
fn get_device() -> &'static str {
    return "cpu";
}


#[derive(StructOpt)]
struct Cli {
    /// The running mode
    mode: String,
    /// The model used for training
    model: String,
    /// The path to the file to read
    epochs: u32,
    /// The number of running times
    run_num: u32,
    /// Whether using SGD with learning rate only, default is false
    sgd_lr_only: Option<bool>
}


pub fn read_txt_file<T>(file_name:&str) -> impl Iterator<Item = T>
    where T: FromStr, <T as FromStr>::Err : std::fmt::Debug
{
    let file = File::open(file_name).expect("could not open file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(
            |line|
                line.expect("failed to read line from file"))
        .map(
            |line|
                line.parse::<T>().expect(&format!("unable to parse {}", line)))
}


pub fn write_to_file<T: AsRef<std::path::Path>>(dir: T, data:Vec<f64>) -> Result<()> {
    let file = File::create(dir).unwrap();
    let mut file = BufWriter::new(file);
    for value in data {
        writeln!(file, "{:.16}", value)?;
    }
    Ok(())
}


// fn run(dataset_name: &str, out_path: &str, model_name: &str, epochs: u32, run_num: u32) {
//     for cur_run in 0..run_num {
//         let batch_size = 128;
//         let (x_train_batches,
//             y_train_batches,
//             x_test_batches,
//             y_test_batches,
//             y_train,
//             y_test) = dataset_loader::load_dataset(dataset_name, batch_size);
//
//         println!("x_train_batches: {:?}", x_train_batches);
//         println!("y_train_batches: {:?}", y_train_batches);
//         println!("x_test_batches: {:?}", x_test_batches);
//         println!("y_test_batches: {:?}", y_test_batches);
//
//         let mnist_obj = mnist_model::MnistModel::create_model(batch_size);
//         let start = Instant::now();
//         let (training_accs, test_accs, eval_time) = mnist_obj.train(
//             epochs,
//             &x_train_batches, &y_train_batches,
//             &x_test_batches, &y_train, &y_test
//         );
//         let total_time = start.elapsed().as_secs_f64();
//         let train_time = total_time - eval_time;
//         println!("total_time: {}, train: {}, eval: {}", total_time, train_time, eval_time);
//         write_to_file(format!("{}/testing_errors_{}.txt", out_path, cur_run), test_accs).unwrap();
//         write_to_file(format!("{}/training_errors_{}.txt", out_path, cur_run), training_accs).unwrap();
//         let file = File::create(format!("{}/time_cost_gpu_{}.txt", out_path, cur_run)).unwrap();
//         let mut file = BufWriter::new(file);
//         let test_acc = mnist_obj.evaluate(&x_test_batches, &y_test);
//         let train_acc = mnist_obj.evaluate(&x_train_batches, &y_train);
//         writeln!(file, "{:.16}", total_time).unwrap();
//         writeln!(file, "{:.16}", eval_time).unwrap();
//         writeln!(file, "{:.16}", train_time).unwrap();
//         writeln!(file, "{:.16}", test_acc).unwrap();
//         writeln!(file, "{:.16}", train_acc).unwrap();
//     }
// }


fn is_accuracy_equal(acc1: f64, acc2: f64, total_num: usize) -> bool {
    return (acc1 * total_num as f64).round() == (acc2 * total_num as f64).round()
}


fn eval_prof(
    model: &SerializedModel, x_batches: &Vec<tf::Tensor<f32>>, y_gt: &Vec<i32>
) -> (f64, Vec<f64>) {
    let mut avg_times = 0.0;
    let mut accs: Vec<f64> = vec![];
    for _ in 0..5 {
        let start = Instant::now();
        let test_accuracy = model.evaluate(&x_batches, &y_gt);
        let duration = start.elapsed();
        avg_times += duration.as_secs_f64();
        accs.push(test_accuracy);
    }
    avg_times /= 5.0;
    return (avg_times, accs)
}


fn deploy_serialization(dataset_name: &str, model_name: &str, model_num: usize, sgd_lr_only: &bool) {
    let (py_out_path, rs_out_path) = match sgd_lr_only {
        true => {
            (format!("{}/{}/py", OUT_LR_ONLY_PATH, model_name), format!("{}/{}/rs", OUT_LR_ONLY_PATH, model_name))
        },
        false => {
            (format!("{}/{}/py", OUT_PATH, model_name), format!("{}/{}/rs", OUT_PATH, model_name))
        }
    };

    let (input_names, output_names) = match model_name {
        "lenet1" => {
            (vec!["conv2d_input", "conv2d_2_input", "conv2d_4_input", "conv2d_6_input", "conv2d_8_input"],
             vec!["dense", "dense_1", "dense_2", "dense_3", "dense_4"])
        },
        "lenet5" => {
            (vec!["conv2d_input", "conv2d_3_input", "conv2d_6_input", "conv2d_9_input", "conv2d_12_input"],
             vec!["dense_1", "dense_3", "dense_5", "dense_7", "dense_9"])
        },
        "vgg16" => {
            (vec!["conv2d_input", "conv2d_13_input", "conv2d_26_input", "conv2d_39_input", "conv2d_52_input"],
             vec!["dense_2", "dense_5", "dense_8", "dense_11", "dense_14"])
        },
        "lstm" => {
            (vec!["embedding_input", "embedding_1_input", "embedding_2_input", "embedding_3_input", "embedding_4_input"],
             vec!["dense", "dense_1", "dense_2", "dense_3", "dense_4"])
        },
        "gru" => {
            (vec!["embedding_input", "embedding_1_input", "embedding_2_input", "embedding_3_input", "embedding_4_input"],
             vec!["dense", "dense_1", "dense_2", "dense_3", "dense_4"])
        },
        _ => {panic!("invalid model name")}
    };

    let batch_size = 128;
    let (x_train_batches,
        y_train_batches,
        x_test_batches,
        y_test_batches,
        y_train,
        y_test) = dataset_loader::load_dataset(dataset_name, batch_size);

    let mut test_times: Vec<f64> = vec![];
    let mut test_accs: Vec<f64> = vec![];
    let mut test_same_accs: Vec<bool> = vec![];
    let mut train_times: Vec<f64> = vec![];
    let mut train_accs: Vec<f64> = vec![];
    let mut train_same_accs: Vec<bool> = vec![];

    for i in 0..model_num {
        println!("deploying evaluation by serialization {:?}", i);
        let export_dir = format!("{}/{}-{}_{}", py_out_path, dataset_name, model_name, i);
        let model = SerializedModel::from_dir(
            &export_dir, input_names[i], output_names[i]
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string())).unwrap();

        // test set
        let testing_accs_gt: Vec<f64> = read_txt_file(
            &format!("{}/testing_errors_{}.txt", py_out_path, i)
        ).collect();
        let test_acc0 = model.evaluate(&x_test_batches, &y_test);
        test_accs.push(test_acc0);
        let (test_eval_time, test_eval_accs) = eval_prof(&model, &x_test_batches, &y_test);
        test_times.push(test_eval_time);
        for acc in test_eval_accs {
            assert!(is_accuracy_equal(acc, test_acc0, y_test.len()))
        }
        let test_set_same_acc = is_accuracy_equal(
            test_acc0, testing_accs_gt[testing_accs_gt.len() - 1], y_test.len()
        );
        test_same_accs.push(test_set_same_acc);

        // train set
        let training_accs_gt: Vec<f64> = read_txt_file(
            &format!("{}/training_errors_{}.txt", py_out_path, i)
        ).collect();
        let train_acc0 = model.evaluate(&x_train_batches, &y_train);
        train_accs.push(train_acc0);
        let (train_eval_time, train_eval_accs) = eval_prof(&model, &x_train_batches, &y_train);
        train_times.push(train_eval_time);
        for acc in train_eval_accs {
            assert!(is_accuracy_equal(acc, train_acc0, y_train.len()))
        }
        let train_set_same_acc = is_accuracy_equal(
            train_acc0, training_accs_gt[training_accs_gt.len() - 1],y_train.len()
        );
        train_same_accs.push(train_set_same_acc);
    }
    // save results
    let file = File::create(
        &format!("{}/deploy_eval_{}_serialization.txt", rs_out_path, get_device()),
    ).unwrap();
    let mut file = BufWriter::new(file);
    writeln!(file, "Test Eval Average Time: {:.16}", test_times.iter().sum::<f64>() / test_times.len() as f64).unwrap();
    for value in test_times {
        writeln!(file, "{:.16}", value).unwrap();
    }
    writeln!(file, "Test Accuracy Same as Original:").unwrap();
    for idx in 0..test_same_accs.len() {
        writeln!(file, "{:.16}, {:.16}", test_same_accs[idx], test_accs[idx]).unwrap();
    }

    writeln!(file, "Train Eval Average Time: {:.16}", train_times.iter().sum::<f64>() / train_times.len() as f64).unwrap();
    for value in train_times {
        writeln!(file, "{:.16}", value).unwrap();
    }
    writeln!(file, "Train Accuracy Same as Original:").unwrap();
    for idx in 0..train_same_accs.len() {
        writeln!(file, "{:.16}, {:.16}", train_same_accs[idx], train_accs[idx]).unwrap();
    }
}


fn main() {
    // let (x_train_batches,
    //     y_train_batches,
    //     x_test_batches,
    //     y_test_batches,
    //     y_train,
    //     y_test) =  dataset_loader::load_imdb_batches(256);
    // let m = graph_model::GraphModel::from_dir(
    //     "./lenet5_frozen_graph.pb",
    //     "conv2d_input",
    //     "dense_1",
    // ).unwrap();

    let args: Cli = Cli::from_args();
    let sgd_lr_only = match args.sgd_lr_only {
        Some(flag) => {
            if flag {
                true
            } else {
                false
            }
        }
        _ => false
    };
    println!(
        "{:?}ing {:?} on {:?}, epochs {:?}, run {:?}, sgd_lr_only {:?}",
        args.mode, args.model, get_device(), args.epochs, args.run_num, sgd_lr_only
    );

    let dataset_name: &str;
    match args.model.as_str() {
        "lenet1" => {
            dataset_name = "mnist";
        },
        "lenet5" => {
            dataset_name = "mnist";
        },
        "vgg16" => {
            dataset_name = "cifar";
        },
        "resnet20" => {
            dataset_name = "cifar";
        },
        "lstm" => {
            dataset_name = "imdb";
        },
        "gru" => {
            dataset_name = "imdb";
        },
        _ => {
            panic!("model must be one of ['lenet1', 'lenet5', 'vgg16', 'resnet20', 'lstm', 'gru']")
        }
    }

    let out_path = match sgd_lr_only {
        true => format!("{}/{}/rs", OUT_LR_ONLY_PATH, args.model),
        false => format!("{}/{}/rs", OUT_PATH, args.model),
    };
    create_dir_all(&out_path).unwrap();
    match args.mode.as_str() {
        // "train" => {
        //     run(dataset_name, &out_path, &args.model, args.epochs, args.run_num);
        // },
        "deploy" => {
            deploy_serialization(dataset_name, &args.model, args.run_num as usize, &sgd_lr_only);
        },
        _ => {
            panic!("mode must be 'train', or 'deploy'")
        }
    }
}
