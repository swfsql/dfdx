use crate::prelude::*;
use std::fmt::Debug;

/// Calls on [crate::tensor_ops::BroadcastTo].
#[derive(Clone, Copy, Debug, Default, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Broadcast<Dst: Shape, Ax: Debug>(
    #[cfg_attr(feature = "safetensors", serialize)] pub Dst,
    #[cfg_attr(feature = "safetensors", serialize)] pub Ax,
);

impl<S: Shape, Ax: Axes + Debug, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for Broadcast<S, Ax> {
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<Dst: Shape, Ax: Axes + Debug, Input> Module<Input> for Broadcast<Dst, Ax>
where
    Input: BroadcastTo,
    Dst: ReduceShapeTo<<Input as HasShape>::Shape, Ax>,
{
    type Output = <Input as HasShape>::WithShape<Dst>;
    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_broadcast_like(&self.0)
    }
}
