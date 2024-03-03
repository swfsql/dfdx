use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls on [crate::tensor_ops::MaxTo].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct MaxTo<Dst: Shape, Ax: Axes + Debug>(PhantomData<(Dst, Ax)>);

impl<Dst: Shape, Ax: Axes + Debug, Input> Module<Input> for MaxTo<Dst, Ax>
where
    Input: crate::tensor_ops::MaxTo,
    <Input as HasShape>::Shape: ReduceShapeTo<Dst, Ax>,
{
    type Output = <Input as HasShape>::WithShape<Dst>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_max::<Dst, Ax>()
    }
}
