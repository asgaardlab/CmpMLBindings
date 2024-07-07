use std::fs::File;
use std::io::{BufRead, BufReader};
use tensorflow as tf;
use tf::Tensor;
use std::str::FromStr;
use tensorflow::ImportGraphDefOptions;


static MNIST_DATA_PATH: &str = "../../../data/MNIST";
static CIFAR_DATA_PATH: &str = "../../../data/cifar-10-txt";
static IMDB_DATA_PATH: &str = "../../../data/imdb";


fn read_images(path: &str) -> Vec<Vec<f32>> {
    let f = BufReader::new(File::open(path).unwrap());
    let arr: Vec<Vec<f32>> = f.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse::<u8>().unwrap() as f32 / 255.0f32)
            .collect())
        .collect();
    arr
}


fn read_labels<T: FromStr>(path: &str) -> Vec<T> {
    let f = BufReader::new(File::open(path).unwrap());
    let arr: Vec<T> = f.lines()
        .map(|number| number.unwrap().parse().ok().expect("parsing error"))
        .collect();
    arr
}


fn read_txt_2d<T: FromStr>(path: &str) -> Vec<Vec<T>> {
    let f = BufReader::new(File::open(path).unwrap());
    let arr: Vec<Vec<T>> = f.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse::<T>().ok().expect("parsing error"))
            .collect())
        .collect();
    arr
}


// pub enum XTensor {
//     Float32(Vec<Tensor<f32>>),
//     Int32(Vec<Tensor<i32>>),
// }


pub fn load_dataset(dataset_name: &str, batch_size: usize)
    -> (Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<i32>, Vec<i32>) {
    match dataset_name {
        "mnist" => load_mnist_batches(batch_size),
        "cifar" => load_cifar_batches(batch_size),
        "imdb" => load_imdb_batches(batch_size),
        _ => panic!("wrong dataset_name"),
    }

}


pub fn load_mnist_batches(batch_size: usize) -> (Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<i32>, Vec<i32>) {
    let y_train = read_labels(&format!("{}/y_train_padded.txt", MNIST_DATA_PATH));
    let x_train = read_images(&format!("{}/x_train_padded.txt", MNIST_DATA_PATH));
    let y_test = read_labels(&format!("{}/y_test_padded.txt", MNIST_DATA_PATH));
    let x_test = read_images(&format!("{}/x_test_padded.txt", MNIST_DATA_PATH));

    // println!("x_train: {:?} x {:?}", x_train.len(), x_train[0].len());
    // println!("y_train: {:?}", y_train.len());
    // println!("x_test: {:?} x {:?}", x_test.len(), x_test[0].len());
    // println!("y_test: {:?}", y_test.len());
    fn x_to_batches(data: Vec<Vec<f32>>, batch_size:usize) -> Vec<Tensor<f32>> {
        data.chunks(batch_size)
            .map(
                |x|
                    Tensor::new(
                        &[x.len() as u64, 32, 32, 1]
                    ).with_values(
                        &x.to_vec().into_iter().flatten().collect::<Vec<f32>>())
                        .unwrap())
            .collect()
    }
    fn y_to_batches(data: Vec<i32>, batch_size:usize) -> Vec<Tensor<i32>> {
        data.chunks(batch_size)
            .map(
                |x|
                    Tensor::new(
                        &[x.len() as u64]
                    ).with_values(&x.to_vec())
                        .unwrap())
            .collect()
    }
    let x_train_batches = x_to_batches(x_train.clone(), batch_size);
    let y_train_batches = y_to_batches(y_train.clone(), batch_size);
    let x_test_batches = x_to_batches(x_test.clone(), batch_size);
    let y_test_batches = y_to_batches(y_test.clone(), batch_size);
    (x_train_batches, y_train_batches, x_test_batches, y_test_batches, y_train, y_test)
}


pub fn load_cifar_batches(batch_size: usize) -> (Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<i32>, Vec<i32>) {
    let mut x_train = read_images(&format!("{}/x_train_0.txt", CIFAR_DATA_PATH));
    for i in 1..5 {
        x_train.extend(read_images(&format!("{}/x_train_{}.txt", CIFAR_DATA_PATH, i)));
    }
    let y_train = read_labels::<i32>(&format!("{}/y_train.txt", CIFAR_DATA_PATH));
    let y_test = read_labels::<i32>(&format!("{}/y_test.txt", CIFAR_DATA_PATH));
    let x_test = read_images(&format!("{}/x_test.txt", CIFAR_DATA_PATH));

    // println!("x_train: {:?} x {:?}", x_train.len(), x_train[0].len());
    // println!("y_train: {:?}", y_train.len());
    // println!("x_test: {:?} x {:?}", x_test.len(), x_test[0].len());
    // println!("y_test: {:?}", y_test.len());
    fn x_to_batches(data: Vec<Vec<f32>>, batch_size:usize) -> Vec<Tensor<f32>> {
        data.chunks(batch_size)
            .map(
                |x|
                    Tensor::new(
                        &[x.len() as u64, 32, 32, 3]
                    ).with_values(
                        &x.to_vec().into_iter().flatten().collect::<Vec<f32>>())
                        .unwrap())
            .collect()
    }
    fn y_to_batches(data: Vec<i32>, batch_size:usize) -> Vec<Tensor<i32>> {
        data.chunks(batch_size)
            .map(
                |x|
                    Tensor::new(
                        &[x.len() as u64]
                    ).with_values(&x.to_vec())
                        .unwrap())
            .collect()
    }
    let x_train_batches = x_to_batches(x_train.clone(), batch_size);
    let y_train_batches = y_to_batches(y_train.clone(), batch_size);
    let x_test_batches = x_to_batches(x_test.clone(), batch_size);
    let y_test_batches = y_to_batches(y_test.clone(), batch_size);
    (x_train_batches, y_train_batches, x_test_batches, y_test_batches, y_train, y_test)
}


pub fn load_imdb_batches(batch_size: usize) -> (Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<Tensor<f32>>, Vec<Tensor<i32>>, Vec<i32>, Vec<i32>) {
    let x_train = read_txt_2d::<f32>(&format!("{}/x_train.txt", IMDB_DATA_PATH));
    let y_train = read_labels::<i32>(&format!("{}/y_train.txt", IMDB_DATA_PATH));
    let x_test = read_txt_2d::<f32>(&format!("{}/x_test.txt", IMDB_DATA_PATH));
    let y_test = read_labels::<i32>(&format!("{}/y_test.txt", IMDB_DATA_PATH));

    println!("x_train: {:?} x {:?}", x_train.len(), x_train[0].len());
    println!("y_train: {:?}", y_train.len());
    println!("x_test: {:?} x {:?}", x_test.len(), x_test[0].len());
    println!("y_test: {:?}", y_test.len());
    fn x_to_batches(data: Vec<Vec<f32>>, batch_size:usize) -> Vec<Tensor<f32>> {
        data.chunks(batch_size)
            .map(
                |x|
                    Tensor::new(
                        &[x.len() as u64, x[0].len() as u64]
                    ).with_values(
                        &x.to_vec().into_iter().flatten().collect::<Vec<f32>>())
                        .unwrap())
            .collect()
    }
    fn y_to_batches(data: Vec<i32>, batch_size:usize) -> Vec<Tensor<i32>> {
        data.chunks(batch_size)
            .map(
                |x|
                    Tensor::new(
                        &[x.len() as u64]
                    ).with_values(&x.to_vec())
                        .unwrap())
            .collect()
    }
    let x_train_batches = x_to_batches(x_train.clone(), batch_size);
    let y_train_batches = y_to_batches(y_train.clone(), batch_size);
    let x_test_batches = x_to_batches(x_test.clone(), batch_size);
    let y_test_batches = y_to_batches(y_test.clone(), batch_size);
    (x_train_batches, y_train_batches, x_test_batches, y_test_batches, y_train, y_test)
}
