use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls [crate::tensor_ops::StddevTo].
#[derive(Clone, Copy, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct StddevTo<E, Dst: Shape, Ax: Axes + Debug> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub epsilon: E,
    _phantom: PhantomData<(Dst, Ax)>,
}

impl<E: Dtype, Dst: Shape, Ax: Axes + Debug, D: Device<E>> BuildOnDevice<E, D>
    for StddevTo<E, Dst, Ax>
{
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<Dst: Shape, Ax: Axes + Debug, E: Dtype, Input> Module<Input> for StddevTo<E, Dst, Ax>
where
    Input: crate::tensor_ops::StddevTo<E>,
    <Input as HasShape>::Shape: ReduceShapeTo<Dst, Ax>,
    E: Into<f64>,
{
    type Output = <Input as dfdx_core::shapes::HasShape>::WithShape<Dst>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_stddev::<Dst, Ax>(self.epsilon)
    }
}
