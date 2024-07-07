// use std::time::{Instant};
// use tensorflow as tf;
// use tf::{
//     Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
//     DEFAULT_SERVING_SIGNATURE_DEF_KEY, DataType, Scope, Variable, Session, Output,
//     ops
// };
// use anyhow::Result;
// use tensorflow::train::Optimizer;
// use crate::mnist_model;
//
//
// pub struct VGGModel {
//     session: Session,
//     loss_opt: Operation,
//     loss_index: i32,
//     input_op: Operation,
//     input_index: i32,
//     label_op: Operation,
//     label_index: i32,
//     output_op: Operation,
//     output_index: i32,
//     minimize_op: Operation,
// }
//
//
// fn create_conv2d<
//     O0: ::std::convert::Into<Output>
// >(input: O0, weight_shape: [i32; 4], scope: &mut Scope) -> (Operation, Variable, Variable) {
//     let w1_shape = ops::constant(&weight_shape[..], scope).unwrap();
//     let w1_init = ops::RandomStandardNormal::new()
//         .dtype(DataType::Float)
//         .build(w1_shape, scope)
//         .unwrap();
//     let w1 = Variable::builder()
//         .initial_value(w1_init)
//         .data_type(DataType::Float)
//         .shape(weight_shape)
//         .build(&mut scope.with_op_name("conv2d_w"))
//         .unwrap();
//     let b1 = Variable::builder()
//         .const_initial_value(Tensor::<f32>::new(&[weight_shape[3] as u64]))
//         .build(&mut scope.with_op_name("conv2d_b"))
//         .unwrap();
//     let c1 = ops::Conv2D::new()
//         .strides([1, 1, 1, 1])
//         .padding("SAME")
//         .build(input, w1.output().clone(), scope)
//         .unwrap();
//     let c1 = ops::Add::new()
//         .build(c1, b1.output().clone(), scope)
//         .unwrap();
//     let c1 = ops::Tanh::new().build(c1, scope).unwrap();
//     (c1, w1, b1)
// }
//
//
// fn create_linear<
//     O0: ::std::convert::Into<Output>
// >(input: O0, weight_shape: [i32; 2], scope: &mut Scope) -> (Operation, Variable, Variable) {
//     let w_shape = ops::constant(&weight_shape[..], scope).unwrap();
//     let w_init = ops::RandomStandardNormal::new()
//         .dtype(DataType::Float)
//         .build(w_shape, scope)
//         .unwrap();
//     let w = Variable::builder()
//         .initial_value(w_init)
//         .data_type(DataType::Float)
//         .shape(weight_shape)
//         .build(&mut scope.with_op_name("linear_weight"))
//         .unwrap();
//     let b = Variable::builder()
//         .const_initial_value(Tensor::<f32>::new(&[weight_shape[1] as u64]))
//         .build(&mut scope.with_op_name("linear_b"))
//         .unwrap();
//     let la = ops::MatMul::new()
//         .build(input, w.output().clone(), scope)
//         .unwrap();
//     let lb = ops::Add::new()
//         .build(la, b.output().clone(), scope)
//         .unwrap();
//     (lb, w, b)
// }
//
//
//
// impl VGGModel {
//     pub fn conv_bn(in_op: Operation, scope: &mut Scope, c_in: i64, c_out: i64) -> (Operation, Vec<Variable>) {
//         // in_channel, out_channel
//         let conv2d_cfg = nn::ConvConfig { padding: 1, bias: false, ..Default::default() };
//         let mut in_c = c_in;
//         let mut vars: Vec<Variable> = vec![];
//         let (o1, v1, v2) = create_conv2d(in_op, [3, 3, c_in, i], scope);
//
//
//             // let l = seq.len();
//             // seq = seq.add(nn::conv2d(vs / &l.to_string(), in_c, *i, 3, conv2d_cfg));
//             // let l = seq.len();
//             // seq = seq.add(nn::batch_norm2d(vs / &l.to_string(), *i, Default::default()));
//             // seq = seq.add_fn(|x| x.relu());
//             // in_c = *i;
//         seq.add_fn(|x| x.max_pool2d_default(2))
//     }
//
//     pub fn create_model(batch_size: usize) -> Self {
//         let config = vec! [64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0];
//
//         let mut scope = Scope::new_root_scope();
//         let input = ops::Placeholder::new()
//             .dtype(DataType::Float)
//             .shape([batch_size as u64, 32, 32, 3])
//             .build(&mut scope.with_op_name("input"))
//             .unwrap();
//         let label = ops::Placeholder::new()
//             .dtype(DataType::Int32)
//             .shape([batch_size as u64])
//             .build(&mut scope.with_op_name("label"))
//             .unwrap();
//         let label_onehot = ops::OneHot::new()
//             .build(
//                 label.clone(),
//                 ops::constant(10, &mut scope).unwrap(),
//                 ops::constant(1f32, &mut scope).unwrap(),
//                 ops::constant(0f32, &mut scope).unwrap(),
//                 &mut scope)
//             .unwrap();
//
//         let (c1, w1, b1) = create_conv2d(input.clone(), [5, 5, 1, 6], &mut scope);
//         let o1 = ops::MaxPool::new()
//             .ksize([1, 2, 2, 1])
//             .strides([1, 2, 2, 1])
//             .padding("VALID")
//             .build(c1, &mut scope)
//             .unwrap();
//         let (c2, w2, b2) = create_conv2d(o1, [5, 5, 6, 16], &mut scope);
//         // let o2 = ops::AvgPool::new()
//         //     .ksize([1, 2, 2, 1])
//         //     .strides([1, 2, 2, 1])
//         //     .padding("VALID")
//         //     .build(c2, &mut scope)
//         //     .unwrap();
//         let o2 = ops::MaxPool::new()
//             .ksize([1, 2, 2, 1])
//             .strides([1, 2, 2, 1])
//             .padding("VALID")
//             .build(c2, &mut scope)
//             .unwrap();
//         let (o3, w3, b3) = create_conv2d(o2, [5, 5, 16, 120], &mut scope);
//
//         let flatten_shape = ops::constant(&[-1, 120][..], &mut scope).unwrap();
//         let flatten = ops::reshape(o3, flatten_shape, &mut scope).unwrap();
//
//         let (l1, lw1, lb1) = create_linear(flatten, [120, 84], &mut scope);
//         let l1 = ops::Tanh::new().build(l1, &mut scope).unwrap();
//         let (logits, lw2, lb2) = create_linear(l1, [84, 10], &mut scope);
//         // let logits = ops::Softmax::new().build(logits, &mut scope).unwrap();
//
//         // let loss = ops::sparse_softmax_cross_entropy_with_logits(logits.clone(), label.clone(), &mut scope).unwrap();
//         let loss = ops::softmax_cross_entropy_with_logits(logits.clone(), label_onehot, &mut scope).unwrap();
//         let loss = ops::sum(loss, ops::constant(0i32, &mut scope).unwrap(), &mut scope).unwrap();
//         let loss = ops::div(loss, ops::constant(batch_size as f32, &mut scope).unwrap(), &mut scope).unwrap();
//
//         // let optimizer = tf::train::GradientDescentOptimizer::new(
//         //     ops::constant(0.05f32, &mut scope).unwrap()
//         // );
//         let mut optimizer = tf::train::MomentumOptimizer::new();
//         // let mut lr: Tensor<f32> = Tensor::new(&[1]);
//         // lr.set(&[0], 0.05);
//         // let lr_o = ops::constant(0.05f32, &mut scope).unwrap();
//         // let lr_o = ops::constant(Tensor::new(&[1]).with_values(&[0.05f32])?, &mut scope)?;
//         // let mo_o = ops::constant(Tensor::new(&[1]).with_values(&[0.9f32])?, &mut scope)?;
//
//         // let mo_o = ops::constant(0.9f32, &mut scope).unwrap();
//         // let mut mo: Tensor<f32> = Tensor::new(&[1]);
//         // mo.set(&[0], 0.9);
//         // println!("lr: {:?}, mo: {:?}, old: {:?}, {:?}", lr, mo, lr_o, mo_o);
//         optimizer.set_learning_rate(ops::constant(0.05f32, &mut scope).unwrap());
//         optimizer.set_momentum(ops::constant(0.9f32, &mut scope).unwrap());
//         println!("scope {:?}", scope);
//
//         let variables = vec![
//             w1, b1, w2, b2, w3, b3,
//             lw1, lb1, lw2, lb2
//         ];
//         let (minimizer_vars, minimize) = optimizer
//             .minimize(
//                 &mut scope,
//                 loss.clone().into(),
//                 tf::train::MinimizeOptions::default().with_variables(&variables),
//             )
//             .unwrap();
//         let options = SessionOptions::new();
//         let g = scope.graph_mut();
//         let session = Session::new(&options, &g).unwrap();
//
//         let mut run_args = SessionRunArgs::new();
//         for var in &variables {
//             run_args.add_target(&var.initializer());
//         }
//         for var in &minimizer_vars {
//             run_args.add_target(&var.initializer());
//         }
//         session.run(&mut run_args).unwrap();
//
//         Self {
//             session,
//             loss_opt: loss,
//             loss_index: 0,
//             input_op: input,
//             input_index: 0,
//             label_op: label,
//             label_index: 0,
//             output_op: logits,
//             output_index: 0,
//             minimize_op: minimize
//         }
//     }
//
//     fn train_step(&self, i: usize, x_batches: &Vec<Tensor<f32>>, y_batches: &Vec<Tensor<i32>>) -> f32 {
//         let mut run_args = SessionRunArgs::new();
//         run_args.add_target(&self.minimize_op);
//         let loss_token = run_args.request_fetch(&self.loss_opt, self.loss_index);
//         run_args.add_feed(&self.input_op, self.input_index, &x_batches[i]);
//         run_args.add_feed(&self.label_op, self.label_index, &y_batches[i]);
//         self.session.run(&mut run_args).unwrap();
//         let loss_val = run_args.fetch::<f32>(loss_token).unwrap()[0];
//         loss_val
//     }
//
//     pub fn train(
//         &self, epochs: u32,
//         x_train_batches: &Vec<Tensor<f32>>, y_train_batches: &Vec<Tensor<i32>>,
//         x_test_batches: &Vec<Tensor<f32>>, y_train: &Vec<i32>, y_test: &Vec<i32>
//     ) -> (Vec<f64>, Vec<f64>, f64) {
//         let mut test_accs: Vec<f64> = Vec::new();
//         let mut training_accs: Vec<f64> = Vec::new();
//         let mut eval_time = 0.0;
//         for epoch in 0..epochs {
//             println!("epoch: {:?}", epoch);
//             for i in 0..x_train_batches.len() {
//                 self.train_step(i, &x_train_batches, &y_train_batches);
//             }
//             let start = Instant::now();
//             let test_acc = self.evaluate(&x_test_batches, &y_test);
//             let train_acc = self.evaluate(&x_train_batches, &y_train);
//             training_accs.push(train_acc);
//             test_accs.push(test_acc);
//             eval_time += start.elapsed().as_secs_f64();
//         }
//         return (training_accs, test_accs, eval_time);
//     }
//
//     pub fn evaluate(
//         &self,
//         x_batches: &Vec<Tensor<f32>>, y_gt: &Vec<i32>
//     ) -> f64 {
//         mnist_model::evaluate(
//             &self.session, &self.input_op, self.input_index,
//             &self.output_op, self.output_index, x_batches, y_gt
//         )
//     }
// }
//
//
