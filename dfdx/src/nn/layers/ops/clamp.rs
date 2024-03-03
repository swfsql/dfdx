use crate::prelude::*;
use std::fmt::Debug;

/// Calls [crate::tensor_ops::clamp].
#[derive(Clone, Copy, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Clamp<E> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub min: E,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub max: E,
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Clamp<E> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Clamp<E>
where
    E: Into<f64>,
{
    type Output = Tensor<S, E, D, T>;

    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_clamp(self.min, self.max)
    }
}
