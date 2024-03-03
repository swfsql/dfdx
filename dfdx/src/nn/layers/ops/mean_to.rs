use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls on [crate::tensor_ops::MeanTo].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct MeanTo<Dst: Shape, Ax: Axes + Debug>(PhantomData<(Dst, Ax)>);

impl<Dst: Shape, Ax: Axes + Debug, Input> Module<Input> for MeanTo<Dst, Ax>
where
    Input: dfdx_core::tensor_ops::MeanTo,
    <Input as HasShape>::Shape: ReduceShapeTo<Dst, Ax>,
{
    type Output = <Input as HasShape>::WithShape<Dst>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_mean::<Dst, Ax>()
    }
}
