use anyhow::Result;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter, Write, prelude::*};
use std::time::{Instant};
use structopt::StructOpt;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor, vision::dataset::Dataset, Kind, data::Iter2};
use tch::nn::{Optimizer, Sgd, VarStore, Adam};
use std::str::FromStr;
use std::path::Path;


mod mnist_lenet;
mod cifar_vgg;
mod cifar_resnet;
mod imdb_rnns;


static CIFAR_DATA_PATH: &str = "../../../data/cifar-10-batches-bin";
static IMDB_DATA_PATH: &str = "../../../data/imdb";
static MNIST_DATA_PATH: &str = "../../../data/MNIST/raw";
static OUT_PATH: &str = "../../../out/pytorch";
static OUT_LR_ONLY_PATH: &str = "../../../out/pytorch_lr_only";
static SEEDS_FILE_PATH: &str = "../../../random_seeds.txt";


#[derive(StructOpt)]
struct Cli {
    /// The running mode
    mode: String,
    /// The model used for training
    model: String,
    /// The device used: cpu or gpu
    device: String,
    /// The path to the file to read
    epochs: u32,
    /// The number of running times
    run_num: u32,
    /// The index for gpu, default is 0
    gpu_device_index: Option<usize>,
    /// Whether using SGD with learning rate only, default is false
    sgd_lr_only: Option<bool>
}


pub fn read_txt_file_2d(file_name: &str) -> Vec<Vec<i32>>
{
    let file = File::open(file_name).expect("could not open file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect())
        .collect()
}


pub fn read_txt_file<T>(file_name: &str) -> impl Iterator<Item = T>
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
        writeln!(file, "{:.16}", value).unwrap();
    }
    Ok(())
}


enum Model {
    LeNet1(mnist_lenet::LeNet1),
    LeNet5(mnist_lenet::LeNet5),
    VGG16(cifar_vgg::VGG16),
    ResNet20(cifar_resnet::ResNet20),
    RNNModel(imdb_rnns::RNNModel),
    CModule(tch::CModule)
}


impl Model {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        match self {
            Model::VGG16(c) => {
                c.forward_t(xs, train)
            },
            Model::LeNet5(c) => {
                c.forward_t(xs, train)
            },
            Model::LeNet1(c) => {
                c.forward_t(xs, train)
            },
            Model::ResNet20(c) => {
                c.forward_t(xs, train)
            },
            Model::RNNModel(c) => {
                c.forward_t(xs, train)
            },
            Model::CModule(c) => {
                c.forward_t(xs, train)
            },
        }
    }

    fn batch_accuracy_for_logits(
        &self, dataset_name: &DatasetName, dataset: &tch::vision::dataset::Dataset, d: Device, batch_size: i64, train: bool
    ) -> f64 {
        match self {
            Model::VGG16(c) => {
                if train {
                    c.batch_accuracy_for_logits(&dataset.train_images, &dataset.train_labels, d, batch_size)
                } else {
                    c.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, d, batch_size)
                }
            },
            Model::LeNet1(c) => {
                if train {
                    c.batch_accuracy_for_logits(&dataset.train_images, &dataset.train_labels, d, batch_size)
                } else {
                    c.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, d, batch_size)
                }
            },
            Model::LeNet5(c) => {
                if train {
                    c.batch_accuracy_for_logits(&dataset.train_images, &dataset.train_labels, d, batch_size)
                } else {
                    c.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, d, batch_size)
                }
            },
            Model::ResNet20(c) => {
                if train {
                    c.batch_accuracy_for_logits(&dataset.train_images, &dataset.train_labels, d, batch_size)
                } else {
                    c.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, d, batch_size)
                }
            },
            Model::RNNModel(c) => {
                let mut sum_accuracy = 0f64;
                let mut sample_count = 0f64;
                let mut iter_data = match train {
                    true => Iter2::new(&dataset.train_images, &dataset.train_labels, batch_size),
                    false => Iter2::new(&dataset.test_images, &dataset.test_labels, batch_size)
                };
                let _no_grad = tch::no_grad_guard();
                for (xs, ys) in iter_data.return_smaller_last_batch() {
                    let pred = c
                        .forward_t(&xs.to_device(d), false);
                    let acc = pred.round()
                        .eq_tensor(&ys.to_device(d))
                        .to_kind(Kind::Float)
                        .mean(Kind::Float);
                    let size = xs.size()[0] as f64;
                    sum_accuracy += f64::from(&acc) * size;
                    sample_count += size;
                }
                sum_accuracy / sample_count
            },
            Model::CModule(c) => {
                let mut sum_accuracy = 0f64;
                let mut sample_count = 0f64;
                let mut iter_data = match train {
                    true => dataset.train_iter(batch_size),
                    false => dataset.test_iter(batch_size)
                };
                let forward_size: Vec<i64> = match dataset_name {
                    DatasetName::MNIST => vec![-1, 1, 28, 28],
                    DatasetName::CIFAR => vec![-1, 3, 32, 32],
                    DatasetName::IMDb => vec![-1, imdb_rnns::EMBEDDING_VEC_LEN],
                };
                let forward_size_boxed = forward_size.into_boxed_slice();
                for (xs, ys) in iter_data.to_device(d).return_smaller_last_batch() {
                    // let acc = c
                    //     .forward_ts(&[xs.view_(&*forward_size_boxed)])
                    //     .unwrap().accuracy_for_logits(&ys);
                    let pred = c
                        .forward_ts(&[xs.view_(&*forward_size_boxed)]).unwrap();

                    let acc = match dataset_name {
                        DatasetName::IMDb => {
                            pred.round()
                                .eq_tensor(&ys)
                                .to_kind(Kind::Float)
                                .mean(Kind::Float)
                        },
                        _ => {
                            pred.accuracy_for_logits(&ys)
                        }
                    };

                    let size = xs.size()[0] as f64;
                    sum_accuracy += f64::from(&acc) * size;
                    sample_count += size;
                }
                sum_accuracy / sample_count
            },
        }
    }
}


enum Opt {
    SGD(Optimizer<Sgd>),
    Adam(Optimizer<Adam>),
}


impl Opt {
    fn set_lr(&mut self, lr: f64) {
        match self {
            Opt::SGD(o) => {
                o.set_lr(lr)
            },
            Opt::Adam(o) => {
                o.set_lr(lr)
            },
        }
    }

    fn set_momentum(&mut self, m: f64) {
        match self {
            Opt::SGD(o) => {
                o.set_momentum(m)
            },
            Opt::Adam(o) => {
                o.set_momentum(m)
            },
        }
    }

    fn set_weight_decay(&mut self, weight_decay: f64) {
        match self {
            Opt::SGD(o) => {
                o.set_weight_decay(weight_decay)
            },
            Opt::Adam(o) => {
                o.set_weight_decay(weight_decay)
            },
        }
    }

    fn backward_step(&mut self, loss: &Tensor) {
        match self {
            Opt::SGD(o) => {
                o.backward_step(loss)
            },
            Opt::Adam(o) => {
                o.backward_step(loss)
            },
        }
    }
}


enum DatasetName {
    MNIST,
    CIFAR,
    IMDb,
}

impl DatasetName {
    fn as_str(&self) -> &'static str {
        match self {
            DatasetName::MNIST => "mnist",
            DatasetName::CIFAR => "cifar",
            DatasetName::IMDb => "imdb",
        }
    }
}


fn load_imdb_dataset(dir: &str) -> Dataset {
    let x_train_2d = read_txt_file_2d(format!("{}/x_train.txt", dir).as_str());
    let y_train: Vec<i32> = read_txt_file(format!("{}/y_train.txt", dir).as_str()).collect();
    let x_test_2d = read_txt_file_2d(format!("{}/x_test.txt", dir).as_str());
    let y_test: Vec<i32> = read_txt_file(format!("{}/y_test.txt", dir).as_str()).collect();

    let x_train: Vec<i32> = x_train_2d.into_iter().flatten().collect();
    let x_test: Vec<i32> = x_test_2d.into_iter().flatten().collect();

    let x_train = Tensor::of_slice(&x_train)
        .view((i64::from(25000), i64::from(300)))
        .to_kind(Kind::Int);
    let y_train = Tensor::of_slice(&y_train)
        .view((i64::from(25000), i64::from(1)))
        .to_kind(Kind::Float);
    let x_test = Tensor::of_slice(&x_test)
        .view((i64::from(25000), i64::from(300)))
        .to_kind(Kind::Int);
    let y_test = Tensor::of_slice(&y_test)
        .view((i64::from(25000), i64::from(1)))
        .to_kind(Kind::Float);
    Dataset {
        train_images: x_train,
        train_labels: y_train,
        test_images: x_test,
        test_labels: y_test,
        labels: 1
    }
}


fn load_dataset(dataset_name: &DatasetName) -> Dataset {
    match dataset_name {
        DatasetName::MNIST => tch::vision::mnist::load_dir(MNIST_DATA_PATH).unwrap(),
        DatasetName::CIFAR => tch::vision::cifar::load_dir(CIFAR_DATA_PATH).unwrap(),
        DatasetName::IMDb => load_imdb_dataset(IMDB_DATA_PATH),
    }
}


fn get_batch_size(dataset_name: &DatasetName) -> i64 {
    match dataset_name {
        DatasetName::MNIST => 128,
        DatasetName::CIFAR => 128,
        DatasetName::IMDb => 256,
    }
}


// fn create_model(model: &str, vs: &VarStore, sgd_lr_only: &bool) -> (Model, Optimizer) {
fn create_model(model: &str, vs: &VarStore, sgd_lr_only: &bool) -> (Model, Opt) {
    let net: Model;
    let mut opt;
    match model {
        "lenet1" => {
            net = Model::LeNet1(mnist_lenet::LeNet1::new(&vs.root()));
            // opt = nn::Sgd::default().build(&vs, 5e-2).unwrap();
            // if !sgd_lr_only {
            //     opt.set_momentum(9e-1);
            // }
            opt = match sgd_lr_only {
                true => Opt::SGD(nn::Sgd::default().build(&vs, 5e-2).unwrap()),
                false => {
                    let mut temp = Opt::SGD(nn::Sgd::default().build(&vs, 5e-2).unwrap());
                    temp.set_momentum(9e-1);
                    temp
                }
            };
        },
        "lenet5" => {
            net = Model::LeNet5(mnist_lenet::LeNet5::new(&vs.root()));
            // opt = nn::Sgd::default().build(&vs, 5e-2).unwrap();
            // if !sgd_lr_only {
            //     opt.set_momentum(9e-1);
            // }
            opt = match sgd_lr_only {
                true => Opt::SGD(nn::Sgd::default().build(&vs, 5e-2).unwrap()),
                false => {
                    let mut temp = Opt::SGD(nn::Sgd::default().build(&vs, 5e-2).unwrap());
                    temp.set_momentum(9e-1);
                    temp
                }
            };
        },
        "vgg16" => {
            net = Model::VGG16(cifar_vgg::VGG16::new(&vs.root()));
            // opt = nn::Sgd::default().build(&vs, 5e-2).unwrap();
            // if !sgd_lr_only {
            //     opt.set_lr(1e-1);
            //     opt.set_momentum(9e-1);
            //     opt.set_weight_decay(1e-4);
            // }
            opt = match sgd_lr_only {
                true => Opt::SGD(nn::Sgd::default().build(&vs, 5e-2).unwrap()),
                false => {
                    let mut temp = Opt::SGD(nn::Sgd::default().build(&vs, 1e-1).unwrap());
                    temp.set_momentum(9e-1);
                    temp.set_weight_decay(1e-4);
                    temp
                }
            };
        },
        "resnet20" => {
            net = Model::ResNet20(cifar_resnet::ResNet20::new(&vs.root()));
            // opt = nn::Sgd::default().build(&vs, 5e-2).unwrap();
            // if !sgd_lr_only {
            //     opt.set_lr(1e-1);
            //     opt.set_momentum(9e-1);
            //     opt.set_weight_decay(1e-4);
            // }
            opt = match sgd_lr_only {
                true => Opt::SGD(nn::Sgd::default().build(&vs, 5e-2).unwrap()),
                false => {
                    let mut temp = Opt::SGD(nn::Sgd::default().build(&vs, 1e-1).unwrap());
                    temp.set_momentum(9e-1);
                    temp.set_weight_decay(1e-4);
                    temp
                }
            };
        }
        "lstm" => {
            net = Model::RNNModel(imdb_rnns::RNNModel::new(&vs.root(), "lstm"));
            // opt = nn::Adam::default().build(vs, 8e-5).unwrap();
            opt = Opt::Adam(nn::Adam::default().build(vs, 8e-5).unwrap());
        }
        "gru" => {
            net = Model::RNNModel(imdb_rnns::RNNModel::new(&vs.root(), "gru"));
            // opt = nn::Adam::default().build(vs, 3e-4).unwrap();
            opt = Opt::Adam(nn::Adam::default().build(vs, 3e-4).unwrap());
        }
        _ => {
            panic!("model must be one of ['lenet5', 'vgg16', 'resnet20', 'lstm', 'gru']")
        }
    }
    return (net, opt);
}


fn run_sub(device: Device, dataset_name: &DatasetName, out_path: &str, model: &str, epochs: i32, i: i32, sgd_lr_only: &bool, seed: i64) {
    println!("seed: {:?}", seed);
    set_random_seeds(seed);
    let mut test_errors: Vec<f64> = Vec::new();
    let mut training_errors: Vec<f64> = Vec::new();
    let batch_size = get_batch_size(dataset_name);
    let dataset = load_dataset(dataset_name);
    println!("x_train: {:?}", dataset.train_images);
    println!("y_train: {:?}", dataset.train_labels);
    println!("x_test: {:?}", dataset.test_images);
    println!("y_test: {:?}", dataset.test_labels);

    let vs = nn::VarStore::new(device);
    let (net, mut opt) = create_model(model, &vs, sgd_lr_only);
    vs.save(format!("{}/{}-{}_{}_init.pth", out_path, dataset_name.as_str(), model, i)).unwrap();

    let timer0 = Instant::now();
    let mut eval_time = 0.0;
    for epoch in 0..epochs {
        for (bimages, blabels) in dataset.train_iter(batch_size).to_device(vs.device()) {
            let loss = match dataset_name {
                DatasetName::IMDb => {
                    net.forward_t(&bimages, true)
                        .binary_cross_entropy::<Tensor>(&blabels, None, tch::Reduction::Mean)
                },
                _ => {
                    net.forward_t(&bimages, true)
                        .cross_entropy_for_logits(&blabels)
                }
            };
            opt.backward_step(&loss)
        }

        let timer1 = Instant::now();
        let test_accuracy =
            net.batch_accuracy_for_logits(&dataset_name, &dataset,vs.device(), batch_size, false);
        let train_accuracy =
            net.batch_accuracy_for_logits(&dataset_name, &dataset, vs.device(), batch_size, true);
        println!("epoch: {:4} test acc: {:5.2}%, train acc: {:5.2}%", epoch, 100. * test_accuracy, 100. * train_accuracy);
        test_errors.push(test_accuracy);
        training_errors.push(train_accuracy);
        eval_time += timer1.elapsed().as_secs_f64();
    }
    let total_used_time = timer0.elapsed().as_secs_f64();

    println!(
        "Time elapsed in {:?} x {:?} training is: {:?}, eval time: {:?}",
        dataset_name.as_str(), model, total_used_time, eval_time
    );

    let file = File::create(format!("{}/time_cost_{}.txt", out_path, i)).unwrap();
    let mut file = BufWriter::new(file);
    writeln!(file, "{:.16}", total_used_time).unwrap();
    writeln!(file, "{:.16}", eval_time).unwrap();
    writeln!(file, "{:.16}", total_used_time - eval_time).unwrap();
    writeln!(file, "{:?}", seed).unwrap();

    vs.save(format!("{}/{}-{}_{}.pth", out_path, dataset_name.as_str(), model, i)).unwrap();
    write_to_file(format!("{}/testing_errors_{}.txt", out_path, i), test_errors).unwrap();
    write_to_file(format!("{}/training_errors_{}.txt", out_path, i), training_errors).unwrap();
}


fn set_random_seeds(seed: i64) {
    tch::manual_seed(seed);
}

fn run(device: Device, dataset_name: &DatasetName, out_path: &str, model: &str, epochs: i32, run_num: i32, sgd_lr_only: &bool) {
    let seeds: Vec<i64> = read_txt_file(SEEDS_FILE_PATH).collect();
    for i in 0..run_num {
        let seed = seeds[i as usize];
        run_sub(device, dataset_name, out_path, model, epochs, i, sgd_lr_only, seed)
    }
}


fn is_accuracy_equal(acc1: f64, acc2: f64, total_num: i64) -> bool {
    return (acc1 * total_num as f64).round() == (acc2 * total_num as f64).round()
}


fn eval_prof(
    net: &Model, dataset_name: &DatasetName, dataset: &tch::vision::dataset::Dataset, device: Device, batch_size: i64, train: bool
) -> (f64, Vec<f64>) {
    let mut avg_times = 0.0;
    let mut accs: Vec<f64> = vec![];
    for _ in 0..5 {
        let start = Instant::now();
        let test_accuracy = net.batch_accuracy_for_logits(
            dataset_name, dataset, device, batch_size, train
        );
        let duration = start.elapsed();
        avg_times += duration.as_secs_f64();
        accs.push(test_accuracy);
    }
    avg_times /= 5.0;
    return (avg_times, accs)
}


fn deploy_evaluation(device: Device, dataset_name: &DatasetName, model: &str, mode: &str, model_num: i32, sgd_lr_only: &bool, trainMode: bool, scripted: bool) {
    let batch_size = get_batch_size(dataset_name);
    let dataset = load_dataset(dataset_name);
    println!("{:?}", device);
    let (py_out_path, rs_out_path) = match sgd_lr_only {
        true => {
            (format!("{}/{}/py", OUT_LR_ONLY_PATH, model), format!("{}/{}/rs", OUT_LR_ONLY_PATH, model))
        },
        false => {
            (format!("{}/{}/py", OUT_PATH, model), format!("{}/{}/rs", OUT_PATH, model))
        }
    };
    let mut test_times: Vec<f64> = vec![];
    let mut test_accs: Vec<f64> = vec![];
    let mut test_same_accs: Vec<bool> = vec![];
    let mut test_acc_stables: Vec<bool> = vec![];
    let mut train_times: Vec<f64> = vec![];
    let mut train_accs: Vec<f64> = vec![];
    let mut train_same_accs: Vec<bool> = vec![];
    let mut train_acc_stables: Vec<bool> = vec![];
    let mut end_str = match trainMode {
        true => "_train",
        false => ""
    };
    let mut prefix_str = match scripted {
        true => "scripted",
        false => "traced"
    };

    for i in 0..model_num {
        // set_random_seeds();
        println!("deploying evaluation by {:?} {:?}", mode, i);
        let net: Model = match mode {
            "states" => {
                let mut vs = tch::nn::VarStore::new(device);
                let net_temp = create_model(model, &vs, sgd_lr_only).0;

                let tensors = tch::Tensor::read_npz(
                    format!("{}/{}-{}_{}_rust.npz", py_out_path, dataset_name.as_str(), model, i)
                ).unwrap();
                tch::Tensor::save_multi(
                    &tensors,
                    &format!("{}/{}-{}_{}_rust.ot", py_out_path, dataset_name.as_str(), model, i)
                ).unwrap();
                vs.load(
                    &format!("{}/{}-{}_{}_rust.ot", py_out_path, dataset_name.as_str(), model, i)
                ).unwrap();
                net_temp
            },
            "serialization" => {
                let model_path = if device.is_cuda() {
                    let temp = format!("{}/{}_{}-{}_{}_gpu{}.pt", py_out_path, prefix_str, dataset_name.as_str(), model, i, end_str);
                    if Path::new(&temp).exists() {
                        temp
                    } else {
                        format!("{}/{}_{}-{}_{}{}.pt", py_out_path, prefix_str, dataset_name.as_str(), model, i, end_str)
                    }
                } else {
                    format!("{}/{}_{}-{}_{}{}.pt", py_out_path, prefix_str, dataset_name.as_str(), model, i, end_str)
                };
                println!("loaded {}", model_path);
                let mut n = tch::CModule::load_on_device(
                    model_path,
                    device
                ).unwrap();
                n.set_eval();
                Model::CModule(n)
            }
            _ => {
                panic!("mode must be one of ['states', 'serialization']")
            }
        };

        // test set
        let testing_accs_gt: Vec<f64> = read_txt_file(
            &format!("{}/testing_errors_{}.txt", py_out_path, i)
        ).collect();
        let test_acc0 = net.batch_accuracy_for_logits(
            dataset_name, &dataset, device, batch_size, false
        );
        test_accs.push(test_acc0);
        let (test_eval_time, test_eval_accs) = eval_prof(
            &net, dataset_name, &dataset, device, batch_size, false
        );
        test_times.push(test_eval_time);
        let mut test_acc_stable = true;
        for acc in test_eval_accs {
            if (!is_accuracy_equal(acc, test_acc0, dataset.test_labels.size()[0])) {
                test_acc_stable = false;
                break;
            }
        }
        test_acc_stables.push(test_acc_stable);
        let test_set_same_acc = is_accuracy_equal(
            test_acc0, testing_accs_gt[testing_accs_gt.len() - 1],
            dataset.test_labels.size()[0]
        );
        test_same_accs.push(test_set_same_acc);

        // train set
        let training_accs_gt: Vec<f64> = read_txt_file(
            &format!("{}/training_errors_{}.txt", py_out_path, i)
        ).collect();
        let train_acc0 = net.batch_accuracy_for_logits(
            dataset_name, &dataset, device, batch_size, true
        );
        train_accs.push(train_acc0);
        let (train_eval_time, train_eval_accs) = eval_prof(
            &net, dataset_name, &dataset, device, batch_size, true
        );
        train_times.push(train_eval_time);
        let mut train_acc_stable = true;
        for acc in train_eval_accs {
            if (!is_accuracy_equal(acc, train_acc0, dataset.train_labels.size()[0])) {
                train_acc_stable = false;
                break;
            }
        }
        train_acc_stables.push(train_acc_stable);
        let train_set_same_acc = is_accuracy_equal(
            train_acc0, training_accs_gt[training_accs_gt.len() - 1],
            dataset.train_labels.size()[0]
        );
        train_same_accs.push(train_set_same_acc);
    }

    // save results
    let file = File::create(
        &format!("{}/deploy_eval_{:?}_{}{}_{}.txt", rs_out_path, device, mode, end_str, prefix_str),
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

    writeln!(file, "Test Accuracy Stable:").unwrap();
    for value in test_acc_stables {
        writeln!(file, "{:.16}", value).unwrap();
    }
    writeln!(file, "Train Accuracy Stable:").unwrap();
    for value in train_acc_stables {
        writeln!(file, "{:.16}", value).unwrap();
    }
}


fn main() {
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
        "{:?}ing {:?} on {:?}:{:?}, epochs {:?}, run {:?}, sgd_lr_only {:?}",
        args.mode, args.model, args.device, args.gpu_device_index, args.epochs, args.run_num, sgd_lr_only
    );

    let dataset_name: DatasetName;
    match args.model.as_str() {
        "lenet1" => {
            dataset_name = DatasetName::MNIST;
        },
        "lenet5" => {
            dataset_name = DatasetName::MNIST;
        },
        "vgg16" => {
            dataset_name = DatasetName::CIFAR;
        },
        "resnet20" => {
            dataset_name = DatasetName::CIFAR;
        },
        "lstm" => {
            dataset_name = DatasetName::IMDb;
        },
        "gru" => {
            dataset_name = DatasetName::IMDb;
        },
        _ => {
            panic!("model must be one of ['lenet1', 'lenet5', 'vgg16', 'resnet20', 'lstm', 'gru']")
        }
    }

    let device = match args.device.as_str() {
        "gpu" => {
            // let idx: usize = match args.gpu_device_index {
            //     Some(i) => {
            //         i
            //     },
            //     _ => { 0 }
            // };
            // Device::Cuda(idx)
            let d = Device::cuda_if_available();
            assert!(d.is_cuda(), "config cuda error!");
            d
        },
        "cpu" => {
            Device::Cpu
        },
        _ => {
            panic!("device must be 'cpu' or 'gpu'");
        }
    };

    let out_path = match sgd_lr_only {
        true => format!("{}/{}/rs", OUT_LR_ONLY_PATH, args.model),
        false => format!("{}/{}/rs", OUT_PATH, args.model),
    };
    create_dir_all(&out_path).unwrap();
    match args.mode.as_str() {
        "train" => {
            run(
                device, &dataset_name, &out_path, &args.model,
                args.epochs as i32, args.run_num as i32, &sgd_lr_only
            );
        },
        "deploy" => {
            // deploy_evaluation(
            //     device, &dataset_name, &args.model,
            //     "serialization", args.run_num as i32, &sgd_lr_only, true, true
            // );
            deploy_evaluation(
                device, &dataset_name, &args.model,
                "serialization", args.run_num as i32, &sgd_lr_only, false, true
            );
            // deploy_evaluation(
            //     device, &dataset_name, &args.model,
            //     "serialization", args.run_num as i32, &sgd_lr_only, true, false
            // );
            // deploy_evaluation(
            //     device, &dataset_name, &args.model,
            //     "serialization", args.run_num as i32, &sgd_lr_only, false, false
            // );
            deploy_evaluation(
                device, &dataset_name, &args.model,
                "states", args.run_num as i32, &sgd_lr_only, false, true
            );
        },
        _ => {
            panic!("mode must be 'train' or 'deploy'")
        }
    }
}