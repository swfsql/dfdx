use crate::prelude::*;
use std::ops::{BitAnd, BitOr, BitXor, Not as BitNot};

/// Calls on [std::ops::BitAnd], which for booleans is [crate::tensor_ops::bool_and].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct And;

/// Calls on [std::ops::Not], which for booleans is [crate::tensor_ops::bool_not].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Not;

/// Calls on [std::ops::BitOr], which for booleans is [crate::tensor_ops::bool_or].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Or;

/// Calls on [std::ops::BitXor], which for booleans is [crate::tensor_ops::bool_xor].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Xor;

impl<Lhs: BitAnd<Rhs>, Rhs> Module<(Lhs, Rhs)> for And {
    type Output = <Lhs as BitAnd<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        Ok(x.0 & x.1)
    }
}

impl<Input: BitNot> Module<Input> for Not {
    type Output = <Input as BitNot>::Output;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        Ok(!x)
    }
}

impl<Lhs: BitOr<Rhs>, Rhs> Module<(Lhs, Rhs)> for Or {
    type Output = <Lhs as BitOr<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        Ok(x.0 | x.1)
    }
}

impl<Lhs: BitXor<Rhs>, Rhs> Module<(Lhs, Rhs)> for Xor {
    type Output = <Lhs as BitXor<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        Ok(x.0 ^ x.1)
    }
}
