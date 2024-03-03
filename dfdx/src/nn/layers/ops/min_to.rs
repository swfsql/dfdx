use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls on [crate::tensor_ops::MinTo].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct MinTo<Dst: Shape, Ax: Axes + Debug>(PhantomData<(Dst, Ax)>);

impl<Dst: Shape, Ax: Axes + Debug, Input> Module<Input> for MinTo<Dst, Ax>
where
    Input: crate::tensor_ops::MinTo,
    <Input as HasShape>::Shape: ReduceShapeTo<Dst, Ax>,
{
    type Output = <Input as HasShape>::WithShape<Dst>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_min::<Dst, Ax>()
    }
}
