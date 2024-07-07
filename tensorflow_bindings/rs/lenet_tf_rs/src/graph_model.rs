// use std::time::{Instant};
// use std::fs::File;
// use tensorflow as tf;
// use tf::{
//     Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
//     DEFAULT_SERVING_SIGNATURE_DEF_KEY, DataType, Scope, Variable, Session, Output,
//     ops, ImportGraphDefOptions
// };
// use anyhow::Result;
// use tensorflow::train::Optimizer;
// use crate::mnist_model;
// use std::io::Read;
//
//
// pub struct GraphModel {
//     bundle: SavedModelBundle,
//     graph: Graph,
//     input_op: Operation,
//     input_index: i32,
//     output_op: Operation,
//     output_index: i32,
// }
//
//
// impl GraphModel {
//     pub fn from_dir(export_dir: &str, input_name: &str, output_name: &str) -> Result<i32> {
//         const MODEL_TAG: &str = "serve";
//         let mut graph = Graph::new();
//         let bundle =
//             SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, "/tmp/tf_models/saved_model")?;
//         let sig = bundle.meta_graph_def().get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
//         let input_info = sig.get_input(input_name)?;
//         let output_info = sig.get_output(output_name)?;
//         println!("{:?} {:?}", input_info.name(), output_info.name());
//         let input_op = graph.operation_by_name_required(&input_info.name().name)?;
//         let output_op = graph.operation_by_name_required(&output_info.name().name)?;
//         let input_index = input_info.name().index;
//         let output_index = output_info.name().index;
//
//         // let loss = ops::sparse_softmax_cross_entropy_with_logits(logits.clone(), label.clone(), &mut scope).unwrap();
//         let loss = ops::softmax_cross_entropy_with_logits(output_op.clone(), label_onehot, &mut scope).unwrap();
//         let loss = ops::sum(loss, ops::constant(0i32, &mut scope).unwrap(), &mut scope).unwrap();
//         let loss = ops::div(loss, ops::constant(batch_size as f32, &mut scope).unwrap(), &mut scope).unwrap();
//
//         let mut optimizer = tf::train::MomentumOptimizer::new();
//         optimizer.set_learning_rate(ops::constant(0.05f32, &mut scope).unwrap());
//         optimizer.set_momentum(ops::constant(0.9f32, &mut scope).unwrap());
//
//         let (minimizer_vars, minimize) = optimizer
//             .minimize(
//                 &mut scope,
//                 loss.clone().into(),
//                 tf::train::MinimizeOptions::default(),
//             )
//             .unwrap();
//
//         // let mut graph = Graph::new();
//         // let mut proto = Vec::new();
//         // File::open(export_dir).unwrap().read_to_end(&mut proto).unwrap();
//         // // let op_iter = graph.operation_iter();
//         // // let names: Vec<String> = op_iter.map(|x| x.name().unwrap()).collect();
//         // // println!("{:?}", names);
//         // println!("{:?}", graph.graph_def().unwrap());
//         // println!("{:?}", graph.operation_iter());
//         // println!("{:?}", graph.num_functions());
//         // graph.import_graph_def(&proto, &ImportGraphDefOptions::new()).unwrap();
//         // let session = Session::new(&SessionOptions::new(), &graph).unwrap();
//         // let op_x = graph.operation_by_name_required(input_name)?;
//         // // let op_y = graph.operation_by_name_required("output")?;
//         // let op_y = graph.operation_by_name_required(output_name)?;
//
//
//         // let op_init = graph.operation_by_name_required("init")?;
//         // let op_train = graph.operation_by_name_required("SGD")?;
//         // let op_w = graph.operation_by_name_required("w")?;
//         // let op_b = graph.operation_by_name_required("b")?;
//         // let op_file_path = graph.operation_by_name_required("save/Const")?;
//         // let op_save = graph.operation_by_name_required("save/control_dependency")?;
//         Ok(1)
//     }
//
//     // pub fn evaluate(&self, x_batches: &Vec<Tensor<f32>>, y_gt: &Vec<i32>) -> f64 {
//     //     mnist_model::evaluate(
//     //         &self.bundle.session, &self.input_op, self.input_index,
//     //         &self.output_op, self.output_index, x_batches, y_gt
//     //     )
//     // }
//
//     // fn train_step(&self, i: usize, x_batches: &Vec<Tensor<f32>>, y_batches: &Vec<Tensor<i32>>) -> f32 {
//     //     let mut run_args = SessionRunArgs::new();
//     //     run_args.add_target(&self.minimize_op);
//     //     let loss_token = run_args.request_fetch(&self.loss_opt, self.loss_index);
//     //     run_args.add_feed(&self.input_op, self.input_index, &x_batches[i]);
//     //     run_args.add_feed(&self.label_op, self.label_index, &y_batches[i]);
//     //     self.session.run(&mut run_args).unwrap();
//     //     let loss_val = run_args.fetch::<f32>(loss_token).unwrap()[0];
//     //     loss_val
//     // }
//     //
//     // pub fn train(
//     //     &self, epochs: u32,
//     //     x_train_batches: &Vec<Tensor<f32>>, y_train_batches: &Vec<Tensor<i32>>,
//     //     x_test_batches: &Vec<Tensor<f32>>, y_train: &Vec<i32>, y_test: &Vec<i32>
//     // ) -> (Vec<f64>, Vec<f64>, f64) {
//     //     let mut test_accs: Vec<f64> = Vec::new();
//     //     let mut training_accs: Vec<f64> = Vec::new();
//     //     let mut eval_time = 0.0;
//     //     for epoch in 0..epochs {
//     //         println!("epoch: {:?}", epoch);
//     //         for i in 0..x_train_batches.len() {
//     //             self.train_step(i, &x_train_batches, &y_train_batches);
//     //         }
//     //         let start = Instant::now();
//     //         let test_acc = self.evaluate(&x_test_batches, &y_test);
//     //         let train_acc = self.evaluate(&x_train_batches, &y_train);
//     //         training_accs.push(train_acc);
//     //         test_accs.push(test_acc);
//     //         eval_time += start.elapsed().as_secs_f64();
//     //     }
//     //     return (training_accs, test_accs, eval_time);
//     // }
//
// }