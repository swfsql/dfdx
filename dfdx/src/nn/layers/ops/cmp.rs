use crate::prelude::*;

/// Calls on [crate::tensor_ops::TryEq], which for booleans is [crate::tensor_ops::eq].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Eq;

/// Calls on [crate::tensor_ops::TryNe], which for booleans is [crate::tensor_ops::ne].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Ne;

/// Calls on [crate::tensor_ops::TryGt], which for booleans is [crate::tensor_ops::gt].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Gt;

/// Calls on [crate::tensor_ops::TryGe], which for booleans is [crate::tensor_ops::ge].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Ge;

/// Calls on [crate::tensor_ops::TryLt], which for booleans is [crate::tensor_ops::lt].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Lt;

/// Calls on [crate::tensor_ops::TryLe], which for booleans is [crate::tensor_ops::le].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Le;

impl<Lhs: TryEq<Rhs>, Rhs> Module<(Lhs, Rhs)> for Eq {
    type Output = <Lhs as TryEq<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_eq(x.1)
    }
}

impl<Lhs: TryNe<Rhs>, Rhs> Module<(Lhs, Rhs)> for Ne {
    type Output = <Lhs as TryNe<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_ne(x.1)
    }
}

impl<Lhs: TryGt<Rhs>, Rhs> Module<(Lhs, Rhs)> for Gt {
    type Output = <Lhs as TryGt<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_gt(x.1)
    }
}

impl<Lhs: TryGe<Rhs>, Rhs> Module<(Lhs, Rhs)> for Ge {
    type Output = <Lhs as TryGe<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_ge(x.1)
    }
}

impl<Lhs: TryLt<Rhs>, Rhs> Module<(Lhs, Rhs)> for Lt {
    type Output = <Lhs as TryLt<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_lt(x.1)
    }
}

impl<Lhs: TryLe<Rhs>, Rhs> Module<(Lhs, Rhs)> for Le {
    type Output = <Lhs as TryLe<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_le(x.1)
    }
}
