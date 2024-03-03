use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls [crate::tensor_ops::VarTo].
#[derive(Clone, Copy, Debug, CustomModule)]
pub struct VarTo<Dst: Shape, Ax: Axes + Debug>(PhantomData<(Dst, Ax)>);

impl<Dst: Shape, Ax: Axes + Debug, Input> Module<Input> for VarTo<Dst, Ax>
where
    Input: crate::tensor_ops::VarTo,
    <Input as HasShape>::Shape: ReduceShapeTo<Dst, Ax>,
{
    type Output = <Input as dfdx_core::shapes::HasShape>::WithShape<Dst>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_var::<Dst, Ax>()
    }
}
