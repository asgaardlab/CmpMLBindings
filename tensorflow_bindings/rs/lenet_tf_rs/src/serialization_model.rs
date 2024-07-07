use tensorflow as tf;
use tf::{
    Graph, Operation, SavedModelBundle, SessionOptions, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY
};
use anyhow::Result;

use crate::mnist_model;
use crate::dataset_loader;

pub struct SerializedModel {
    bundle: SavedModelBundle,
    input_op: Operation,
    input_index: i32,
    output_op: Operation,
    output_index: i32,
}


impl SerializedModel {
    pub fn from_dir(export_dir: &str, input_name: &str, output_name: &str) -> Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;

        let sig = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
        let input_info = sig.get_input(input_name)?;
        let output_info = sig.get_output(output_name)?;
        println!("{} {}", input_info.name().name, output_info.name().name);
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let output_op = graph.operation_by_name_required(&output_info.name().name)?;
        let input_index = input_info.name().index;
        let output_index = output_info.name().index;

        Ok(Self {
            bundle,
            input_op,
            input_index,
            output_op,
            output_index,
        })
    }

    pub fn evaluate(&self, x_batches: &Vec<Tensor<f32>>, y_gt: &Vec<i32>) -> f64 {
        mnist_model::evaluate(
            &self.bundle.session, &self.input_op, self.input_index,
            &self.output_op, self.output_index, x_batches, y_gt
        )
    }
}


// fn evaluate_model(
//     session: &Session,
//     input_op: &Operation, input_index: i32,
//     output_op: &Operation, output_index: i32,
//     x_batches: &Vec<Tensor<i32>>, y_gt: &Vec<i32>
// ) -> f64 {
//     let mut predict_res: Vec<i32> = vec![];
//     for i in 0..x_batches.len() {
//         let mut run_args = SessionRunArgs::new();
//         run_args.add_feed(input_op, input_index, &x_batches[i]);
//         // run_args.add_feed(&input, 0, &x_batches[i]);
//         // run_args.add_feed(&label, 0, &y_batches[i]);
//         let logits_output = run_args.request_fetch(output_op, output_index);
//         session.run(&mut run_args).unwrap();
//         let output: Tensor<f32> = run_args.fetch(logits_output).unwrap();
//         // println!("output.shape(): {:?}", output.shape());
//         let output_max: Vec<(usize, f32)> = output.chunks(10)
//             .map(|out| {
//                 // println!("{:?}", out);
//                 let res = out
//                     .iter()
//                     .enumerate()
//                     .fold((0, out[0]), |(idx_max, val_max), (idx, val)| {
//                         if &val_max > val {
//                             (idx_max, val_max)
//                         } else {
//                             (idx, *val)
//                         }
//                     });
//                 res
//             }
//             ).collect::<Vec<(usize, f32)>>();
//         // for (idx, value) in output_max.clone() {
//         //     println!("{:?}, {:?}", idx, value);
//         // }
//         let idxs: Vec<i32> = output_max.iter().map(|(idx, _)| *idx as i32).collect();
//         predict_res.extend(idxs);
//     }
//     // println!("predict_res.len(): {:?}", predict_res.len());
//     let mut correct = 0;
//     for (pred, gt) in predict_res.iter().zip(y_gt.iter()) {
//         if *pred == *gt {
//             correct += 1;
//         }
//     }
//     let accuracy = correct as f64 / predict_res.len() as f64;
//     println!("correct: {:?}, total: {:?}, accuracy: {:?}", correct, predict_res.len(), accuracy);
//     accuracy
// }
