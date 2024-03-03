use crate::prelude::*;
use std::fmt::Debug;

/// Calls [crate::tensor_ops::powi].
#[derive(Clone, Copy, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Powi(#[cfg_attr(feature = "safetensors", serialize)] pub i32);

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Powi {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Powi {
    type Output = Tensor<S, E, D, T>;

    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_powi(self.0)
    }
}

/// Calls [crate::tensor_ops::powf].
#[derive(Clone, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Powf<E: Dtype>(#[cfg_attr(feature = "safetensors", serialize)] E);

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Powf<E> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(self.clone())
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Powf<E>
where
    E: Into<f64>,
{
    type Output = Tensor<S, E, D, T>;

    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_powf(self.0)
    }
}
