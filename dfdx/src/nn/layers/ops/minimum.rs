use crate::prelude::*;
use std::fmt::Debug;

/// Calls [crate::tensor_ops::minimum].
#[derive(Clone, Copy, Debug, CustomModule)]
pub struct Minimum;

impl<S: Shape, E: Dtype, D: Device<E>, Lt: Tape<E, D>, Rt: Tape<E, D>>
    Module<(Tensor<S, E, D, Lt>, Tensor<S, E, D, Rt>)> for Minimum
where
    Lt: Merge<Rt>,
{
    type Output = Tensor<S, E, D, Lt>;

    fn try_forward(
        &self,
        x: (Tensor<S, E, D, Lt>, Tensor<S, E, D, Rt>),
    ) -> Result<Self::Output, Error> {
        x.0.try_minimum(x.1)
    }
}
