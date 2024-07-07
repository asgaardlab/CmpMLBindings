use tch::{nn, nn::ModuleT, nn::Conv2D, Tensor};
use tch::nn::{SequentialT};


fn conv_3x3(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride, padding: 1, bias: false, ..Default::default() };
    nn::conv2d(&p, c_in, c_out, 3, conv2d_cfg)
}

fn conv_1x1(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride, bias: false, ..Default::default() };
    nn::conv2d(&p, c_in, c_out, 1, conv2d_cfg)
}

// fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
//     if stride != 1 || c_in != c_out {
//         nn::seq_t().add(
//             conv_1x1(&p / "0", c_in, c_out, stride)
//         ).add(
//             nn::batch_norm2d(&p / "1", c_out,Default::default())
//         )
//     } else {
//         nn::seq_t()
//     }
// }
//
// fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
//     let conv1 = conv_3x3(&p / "conv1", c_in, c_out, stride);
//     let bn1 = nn::batch_norm2d(&p / "bn1", c_out, Default::default());
//     let conv2 = conv_3x3(&p / "conv2", c_out, c_out, 1);
//     let bn2 = nn::batch_norm2d(&p / "bn2", c_out, Default::default());
//     let downsample = downsample(&p / "downsample", c_in, c_out, stride);
//     nn::func_t(move |xs, train| {
//         let ys = xs.apply(&conv1).apply_t(&bn1, train).relu().apply(&conv2).apply_t(&bn2, train);
//         (xs.apply_t(&downsample, train) + ys).relu()
//     })
// }

// fn basic_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> SequentialT {
//     let mut layer = nn::seq_t().add(basic_block(&p / "0", c_in, c_out, stride));
//     layer = layer.add(basic_block(&p / "1", c_out, c_out, 1));
//     layer = layer.add(basic_block(&p / "2", c_out, c_out, 1));
//     layer
// }

fn basic_layer(p: &nn::Path, c_in: i64, c_out: i64, downsample: bool) -> SequentialT {
    let b0 = p / "0";
    let b1 = p / "1";
    let b2 = p / "2";
    let mut layer = nn::seq_t().add(Block::new(&b0, c_in, c_out, downsample));
    layer = layer.add(Block::new(&b1, c_out, c_out, false));
    layer = layer.add(Block::new(&b2, c_out, c_out, false));
    layer
}



#[derive(Debug)]
pub struct ResNet20 {
    pre: SequentialT,
    stack1: SequentialT,
    stack2: SequentialT,
    stack3: SequentialT,
    fcs: SequentialT,
}


impl ResNet20 {
    pub fn new(vs: &nn::Path) -> ResNet20 {
        let nclasses = 10;
        let conv1 = conv_3x3(vs / "conv1", 3, 16, 1);
        let bn1 = nn::batch_norm2d(vs / "bn1", 16, Default::default());
        let pre = nn::seq_t()
            .add(conv1)
            .add(bn1)
            .add_fn(|x| x.relu());

        let s1 = vs / "stack1";
        let s2 = vs / "stack2";
        let s3 = vs / "stack3";

        let stack1 = basic_layer(&s1, 16, 16, false);
        let stack2 = basic_layer(&s2, 16, 32, true);
        let stack3 = basic_layer(&s3, 32, 64, true);

        let fc = nn::linear(vs / "fc", 64, nclasses, Default::default());
        let fcs = nn::seq_t()
            .add_fn(|x| x.max_pool2d_default(8))
            // .add_fn(|x| x.avg_pool2d_default(8))
            // .add_fn(|x| x.adaptive_avg_pool2d(&[1, 1]))
            // .add_fn(|x| x.avg_pool2d(&[8, 8], &[8, 8], &[0, 0], false, true, 8))
            .add_fn(|x| x.relu())
            .add_fn(|x| x.flat_view())
            .add(fc);
        ResNet20 {
            pre,
            stack1,
            stack2,
            stack3,
            fcs
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_t(xs, false)
    }
}


impl nn::ModuleT for ResNet20 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.fcs.forward_t(
            &self.stack3.forward_t(
                &self.stack2.forward_t(
                    &self.stack1.forward_t(
                        &self.pre.forward_t(xs, train), train
                    ), train
                ), train
            ), train
        )
    }
}


#[derive(Debug)]
pub struct Block {
    downsample: bool,
    seq: SequentialT,
    down: SequentialT
}


impl Block {
    pub fn new(vs: &nn::Path, c_in: i64, c_out: i64, downsample: bool) -> Block {
        let mut seq = nn::seq_t();
        if downsample {
            seq = seq.add(conv_3x3(vs / "conv1", c_in, c_out, 2));
        } else {
            seq = seq.add(conv_3x3(vs / "conv1", c_in, c_out, 1));
        }
        seq = seq
            .add(nn::batch_norm2d(vs / "bn1", c_out, Default::default()))
            .add_fn(|x| x.relu())
            .add(conv_3x3(vs / "conv2", c_out, c_out, 1))
            .add(nn::batch_norm2d(vs / "bn2", c_out, Default::default()));

        let mut down = nn::seq_t();
        if downsample {
            down = down
                .add(conv_1x1(vs / "downsample/0", c_in, c_out, 2))
                .add(nn::batch_norm2d(vs / "downsample/1", c_out, Default::default()));
        }
        Block {
            downsample,
            seq,
            down
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_t(xs, false)
    }
}


impl nn::ModuleT for Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let out = self.seq.forward_t(xs, train);
        let identity = self.seq.forward_t(xs, train);
        (out + identity).relu()
    }
}

