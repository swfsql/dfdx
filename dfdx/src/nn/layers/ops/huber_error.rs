use crate::prelude::*;
use std::fmt::Debug;

/// Calls [crate::tensor_ops::huber_error].
#[derive(Clone, Copy, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct HuberError<E> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub delta: E,
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for HuberError<E> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, Lt: Tape<E, D>, Rt: Tape<E, D>>
    Module<(Tensor<S, E, D, Lt>, Tensor<S, E, D, Rt>)> for HuberError<E>
where
    Lt: Merge<Rt>,
    E: Into<f64>,
{
    type Output = Tensor<S, E, D, Lt>;

    fn try_forward(
        &self,
        x: (Tensor<S, E, D, Lt>, Tensor<S, E, D, Rt>),
    ) -> Result<Self::Output, Error> {
        x.0.try_huber_error(x.1, self.delta)
    }
}
