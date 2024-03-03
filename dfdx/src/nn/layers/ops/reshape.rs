use crate::prelude::*;

/// Calls on [crate::tensor_ops::ReshapeTo].  
/// Reshapes input tensors to a configured shape.
///
/// Example usage:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// let model: ops::Reshape<Rank2<5, 24>> = Default::default();
/// let x: Tensor<Rank4<5, 4, 3, 2>, f32, _> = dev.sample_normal();
/// let _: Tensor<Rank2<5, 24>, f32, _> = model.forward(x);
/// ```
#[derive(Clone, Copy, Debug, Default, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Reshape<S: Shape>(#[cfg_attr(feature = "safetensors", serialize)] pub S);

impl<S: Shape, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Reshape<S> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<Dst: Shape, Input> Module<Input> for Reshape<Dst>
where
    Input: ReshapeTo,
{
    type Output = <Input as HasShape>::WithShape<Dst>;
    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_reshape_like(&self.0)
    }
}
