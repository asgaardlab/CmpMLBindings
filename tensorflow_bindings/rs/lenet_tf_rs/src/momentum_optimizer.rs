// diy: add following lines to tensorflow/src/train.rs
#[derive(Debug)]
pub struct MomentumOptimizer {
    learning_rate: Option<Output>,
    momentum: Option<Output>,
}

impl Default for MomentumOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MomentumOptimizer {
    /// Creates a new optimizer with default parameters (learning_rate=0.01, momentum=0.0).
    pub fn new() -> Self {
        Self {
            learning_rate: None,
            momentum: None,
        }
    }

    /// Sets the learning rate.  Default is 0.01.
    pub fn set_learning_rate<T: Into<Output>>(&mut self, learning_rate: T) {
        self.learning_rate = Some(learning_rate.into());
    }

    /// Sets momentum.  Default is 0.0.
    pub fn set_momentum<T: Into<Output>>(&mut self, momentum: T) {
        self.momentum = Some(momentum.into());
    }
}


impl Optimizer for MomentumOptimizer {
    fn apply_gradients(
        &self,
        scope: &mut Scope,
        opts: ApplyGradientsOptions,
    ) -> Result<(Vec<Variable>, Operation)> {
        let learning_rate = or_constant(scope, &self.learning_rate, 0.01f32)?;
        let momentum = or_constant(scope, &self.momentum, 0.0f32)?;
        let mut apply_ops = Vec::new();
        let mut variables = Vec::new();
        for (grad, var) in opts.grads_and_vars {
            if let Some(grad) = grad {
                let mut scope = scope.new_sub_scope(&var.name);
                let accum = create_zeros_slot(&mut scope.new_sub_scope("accum"), var, None)?;
                apply_ops.push(ops::apply_momentum(
                    var.output.clone(),
                    accum.output.clone(),
                    learning_rate.clone(),
                    grad.clone(),
                    momentum.clone(),
                    scope,
                )?);
                variables.push(accum.clone());
            }
        }
        let mut no_op = ops::NoOp::new();
        for apply_op in &apply_ops {
            no_op = no_op.add_control_input(apply_op.clone());
        }
        Ok((variables, no_op.build(scope)?))
    }
}