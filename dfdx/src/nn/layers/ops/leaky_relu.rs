use crate::prelude::*;

/// ReLU but maintains a small gradient if the input values are negative.
#[derive(Clone, Copy, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct LeakyReLU(#[cfg_attr(feature = "safetensors", serialize)] pub f64);

impl Default for LeakyReLU {
    fn default() -> Self {
        Self(0.05)
    }
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for LeakyReLU {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for LeakyReLU {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_prelu(E::from_f64(self.0).unwrap())
    }
}
