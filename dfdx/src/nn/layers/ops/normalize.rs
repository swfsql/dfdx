use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls [crate::tensor_ops::normalize()].
#[derive(Clone, Copy, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Normalize<E: Dtype, Ax: Debug> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub epsilon: E,
    _ax: PhantomData<Ax>,
}

impl<Ax: Axes + Debug, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Normalize<E, Ax> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<Ax: Axes + Debug, S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for Normalize<E, Ax>
where
    E: Into<f64>,
{
    type Output = Tensor<S, E, D, T>;

    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_normalize(self.epsilon)
    }
}
