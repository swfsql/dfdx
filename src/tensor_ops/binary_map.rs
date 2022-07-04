use crate::prelude::*;

pub(super) mod add {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }
    pub fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    pub fn dfdy(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
}

pub(super) mod sub {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }
    pub fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    pub fn dfdy(_x: &f32, _y: &f32) -> f32 {
        -1.0
    }
}

pub(super) mod mul {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }
    pub fn dfdx(_x: &f32, y: &f32) -> f32 {
        *y
    }
    pub fn dfdy(x: &f32, _y: &f32) -> f32 {
        *x
    }
}

pub(super) mod div {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x * y.recip()
    }
    pub fn dfdx(_x: &f32, y: &f32) -> f32 {
        y.recip()
    }
    pub fn dfdy(x: &f32, y: &f32) -> f32 {
        (-x) * y.powi(2).recip()
    }
}

pub(super) mod minimum {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x.min(*y)
    }
    pub fn dfdx(x: &f32, y: &f32) -> f32 {
        if x < y {
            1.0
        } else if x > y {
            0.0
        } else {
            0.5
        }
    }

    pub fn dfdy(x: &f32, y: &f32) -> f32 {
        if y < x {
            1.0
        } else if y > x {
            0.0
        } else {
            0.5
        }
    }
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs.
///
/// This is primarily used to implement [add()], [sub()], [mul()], and [div()].
pub(super) fn binary_map<T: Tensor<Dtype = f32>>(
    lhs: T,
    rhs: &T::NoTape,
    f: fn(&f32, &f32) -> f32,
    dfdx: fn(&f32, &f32) -> f32,
    dfdy: fn(&f32, &f32) -> f32,
) -> T {
    let (lhs, mut tape) = lhs.split_tape();

    let mut result = T::NoTape::zeros();
    T::Device::foreach_mrr(result.mut_data(), lhs.data(), rhs.data(), &mut |o, l, r| {
        *o = f(l, r)
    });

    // calculate derivatives
    let mut rhs_deriv: Box<T::Array> = T::Device::zeros();
    T::Device::foreach_mrr(
        rhs_deriv.as_mut(),
        lhs.data(),
        rhs.data(),
        &mut |o, l, r| {
            *o = dfdy(l, r);
        },
    );
    let mut lhs_deriv = lhs;
    T::Device::foreach_mr(lhs_deriv.mut_data(), rhs.data(), &mut |l, r| {
        *l = dfdx(l, r)
    });

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs_deriv, &_result);
        T::Device::foreach_mrr(lhs_grad, lhs_deriv.data(), result_grad, &mut |g, d, r| {
            *g += d * r;
        });
        let (rhs_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        T::Device::foreach_mrr(rhs_grad, rhs_deriv.as_ref(), result_grad, &mut |g, d, r| {
            *g += d * r;
        });
    });
    result.put_tape(tape)
}

/// Apply binary function `f` to `lhs` and `rhs`, where `rhs` is broadcasted `M` times to be the same shape as `lhs`.
/// `dfdx` and `dfdy` are the partial derivatives of f wrt. x and y respectively.
///
/// `f`, `dfdx`, and `dfdy` are all the same type.
///
/// Generics:
/// - `M`: The first dimension of `lhs`.
pub(super) fn binary_map_broadcast_rhs_first<const M: usize, Lhs, Rhs>(
    lhs: Lhs,
    rhs: &Rhs,
    f: fn(&f32, &f32) -> f32,
    dfdx: fn(&f32, &f32) -> f32,
    dfdy: fn(&f32, &f32) -> f32,
) -> Lhs
where
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs: Tensor<Dtype = f32, Array = [Rhs::Array; M]>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    let (lhs, mut tape) = lhs.split_tape();

    let mut result = Lhs::NoTape::zeros();
    for i in 0..M {
        Lhs::Device::foreach_mrr(
            &mut result.mut_data()[i],
            &lhs.data()[i],
            rhs.data(),
            &mut |o, l, r| {
                *o = f(l, r);
            },
        )
    }

    // calculate derivatives
    let mut rhs_deriv: Box<Lhs::Array> = Lhs::Device::zeros();
    let mut lhs_deriv = lhs;
    for i in 0..M {
        Lhs::Device::foreach_mrr(
            &mut rhs_deriv[i],
            &lhs_deriv.data()[i],
            rhs.data(),
            &mut |o, l, r| {
                *o = dfdy(l, r);
            },
        );
        Lhs::Device::foreach_mr(&mut lhs_deriv.mut_data()[i], rhs.data(), &mut |l, r| {
            *l = dfdx(l, r)
        });
    }

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs_deriv, &_result);
        Lhs::Device::foreach_mrr(lhs_grad, lhs_deriv.data(), result_grad, &mut |g, d, r| {
            *g += d * r;
        });

        let (rhs_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        Lhs::Device::foreach_mr(rhs_deriv.as_mut(), result_grad, &mut |d, r| {
            *d *= r;
        });
        for i in 0..M {
            Rhs::Device::foreach_mr(rhs_grad, &rhs_deriv[i], &mut |g, d| {
                *g += d;
            });
        }
    });
    result.put_tape(tape)
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs. Note that `rhs` has it's last dimension reduced,
/// so therefore it's last dimension is broadcasted to `lhs`'s last dimension.
///
/// This is primarily used to implement [add_broadcast_rhs_last()],
/// [sub_broadcast_rhs_last()], [mul_broadcast_rhs_last()], and [div_broadcast_rhs_last()].
pub(super) fn binary_map_broadcast_rhs_last<T: Tensor<Dtype = f32>>(
    lhs: T,
    mut rhs: <T::LastDimReduced as Tensor>::NoTape,
    f: fn(&f32, &f32) -> f32,
    dfdx: fn(&f32, &f32) -> f32,
    dfdy: fn(&f32, &f32) -> f32,
) -> T {
    let (lhs, mut tape) = lhs.split_tape();

    let mut result = T::NoTape::zeros();
    T::Device::foreach_mrb(
        result.mut_data(),
        lhs.data(),
        Broadcast(rhs.data()),
        &mut |o, l, r| {
            *o = f(l, r);
        },
    );

    // calculate derivatives
    let mut rhs_deriv: Box<T::Array> = T::Device::zeros();
    T::Device::foreach_mrb(
        rhs_deriv.as_mut(),
        lhs.data(),
        Broadcast(rhs.data()),
        &mut |d, l, r| {
            *d = dfdy(l, r);
        },
    );

    let mut lhs_deriv = lhs;
    T::Device::foreach_mb(lhs_deriv.mut_data(), Broadcast(rhs.data()), &mut |l, r| {
        *l = dfdx(l, r)
    });

    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs_deriv, &_result);
        T::Device::foreach_mrr(lhs_grad, lhs_deriv.data(), result_grad, &mut |g, d, r| {
            *g += d * r;
        });

        let (rhs_grad, result_grad) = grads.mut_and_ref(&rhs, &_result);
        T::Device::foreach_mr(rhs_deriv.as_mut(), result_grad, &mut |d, r| {
            *d *= r;
        });
        T::Device::reduce_last_dim_into(rhs_deriv.as_ref(), rhs.mut_data(), &mut |x, y| x + y);
        <T::LastDimReduced as HasDevice>::Device::foreach_mr(rhs_grad, rhs.data(), &mut |g, d| {
            *g += d;
        });
    });
    result.put_tape(tape)
}