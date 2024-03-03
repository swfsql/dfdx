use crate::prelude::*;
use std::fmt::Debug;

/// Calls [crate::tensor_ops::nans_to].
#[derive(Clone, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct NansTo<E: Dtype>(#[cfg_attr(feature = "safetensors", serialize)] E);

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for NansTo<E> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(self.clone())
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for NansTo<E>
where
    E: Into<f64>,
{
    type Output = Tensor<S, E, D, T>;

    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_nans_to(self.0)
    }
}
