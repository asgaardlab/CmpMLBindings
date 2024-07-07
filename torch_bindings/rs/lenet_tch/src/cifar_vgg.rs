use tch::{nn, nn::ModuleT, Tensor};


#[derive(Debug)]
pub struct VGG16 {
    cnn_model: nn::SequentialT,
    fcs: nn::SequentialT,
}


impl VGG16 {
    pub fn new(vs: &nn::Path) -> VGG16 {
        let nclasses = 10;
        let f = vs / "features";
        let c = vs / "classifier";

        let cnn_model = VGG16::create_conv_layer(&f);

        let fcs = nn::seq_t()
            .add_fn(|x| x.flat_view())
            .add(nn::linear(&c / "0", 512, 4096, Default::default()))
            .add_fn(|x| x.relu())
            .add_fn_t(|x, train| x.dropout(0.5, train))
            .add(nn::linear(&c / "3", 4096, 4096, Default::default()))
            .add_fn(|x| x.relu())
            .add_fn_t(|x, train| x.dropout(0.5, train))
            .add(nn::linear(&c / "6", 4096, nclasses, Default::default()));
        VGG16 {
            cnn_model,
            fcs
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_t(xs, false)
    }

    pub fn create_conv_layer(vs: &nn::Path) -> nn::SequentialT {
        let mut seq = nn::seq_t();

        seq = VGG16::conv_bn(vs, seq, 3, vec![64, 64]);
        seq = VGG16::conv_bn(vs, seq, 64, vec![128, 128]);
        seq = VGG16::conv_bn(vs, seq, 128, vec![256, 256, 256]);
        seq = VGG16::conv_bn(vs, seq, 256, vec![512, 512, 512]);
        VGG16::conv_bn(vs, seq, 512, vec![512, 512, 512])
    }

    pub fn conv_bn(vs: &nn::Path, mut seq: nn::SequentialT, c_in: i64, c_out: Vec<i64>) -> nn::SequentialT {
        let conv2d_cfg = nn::ConvConfig { padding: 1, bias: false, ..Default::default() };
        let mut in_c = c_in;
        for i in c_out.iter() {
            let l = seq.len();
            seq = seq.add(nn::conv2d(vs / &l.to_string(), in_c, *i, 3, conv2d_cfg));
            let l = seq.len();
            seq = seq.add(nn::batch_norm2d(vs / &l.to_string(), *i, Default::default()));
            seq = seq.add_fn(|x| x.relu());
            in_c = *i;
        }
        seq.add_fn(|x| x.max_pool2d_default(2))
    }
}


impl nn::ModuleT for VGG16 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.fcs.forward_t(
            &self.cnn_model.forward_t(xs, train)
            , train
        )
    }
}
