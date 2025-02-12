#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]
#![allow(incomplete_features)]

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run this example with `cargo +nightly`.")
}

#[cfg(feature = "nightly")]
fn main() {
    use std::time::Instant;

    use dfdx::prelude::*;

    #[derive(Default, Clone, Sequential)]
    pub struct BasicBlockInternal<const C: usize> {
        conv1: Conv2DConstConfig<C, C, 3, 1, 1>,
        bn1: BatchNorm2DConstConfig<C>,
        relu: ops::ReLU,
        conv2: Conv2DConstConfig<C, C, 3, 1, 1>,
        bn2: BatchNorm2DConstConfig<C>,
    }

    #[derive(Default, Clone, Sequential)]
    pub struct DownsampleA<const C: usize, const D: usize> {
        conv1: Conv2DConstConfig<C, D, 3, 2, 1>,
        bn1: BatchNorm2DConstConfig<D>,
        relu: ops::ReLU,
        conv2: Conv2DConstConfig<D, D, 3, 1, 1>,
        bn2: BatchNorm2DConstConfig<D>,
    }

    #[derive(Default, Clone, Sequential)]
    pub struct DownsampleB<const C: usize, const D: usize> {
        conv1: Conv2DConstConfig<C, D, 1, 2, 0>,
        bn1: BatchNorm2DConstConfig<D>,
    }

    pub type BasicBlock<const C: usize> = ResidualAdd<BasicBlockInternal<C>>;

    pub type Downsample<const C: usize, const D: usize> =
        GeneralizedAdd<DownsampleA<C, D>, DownsampleB<C, D>>;

    #[derive(Default, Clone, Sequential)]
    pub struct Head {
        conv: Conv2DConstConfig<3, 64, 7, 2, 3>,
        bn: BatchNorm2DConstConfig<64>,
        relu: ops::ReLU,
        pool: ops::MaxPool2DConst<3, 2, 1>,
    }

    #[derive(Default, Clone, Sequential)]
    #[built(Resnet18)]
    pub struct Resnet18Config<const NUM_CLASSES: usize> {
        head: Head,
        l1: (BasicBlock<64>, ops::ReLU, BasicBlock<64>, ops::ReLU),
        l2: (Downsample<64, 128>, ops::ReLU, BasicBlock<128>, ops::ReLU),
        l3: (Downsample<128, 256>, ops::ReLU, BasicBlock<256>, ops::ReLU),
        l4: (Downsample<256, 512>, ops::ReLU, BasicBlock<512>, ops::ReLU),
        l5: (AvgPoolGlobal, LinearConstConfig<512, NUM_CLASSES>),
    }

    {
        use dfdx::prelude::*;

        let dev = AutoDevice::default();
        let arch = Resnet18Config::<1000>::default();
        let m: Resnet18<1000, f32, AutoDevice> = dev.build_module::<f32>(arch);

        let x: Tensor<Rank3<3, 224, 224>, f32, _> = dev.sample_normal();
        let start = Instant::now();
        let _: Tensor<Rank1<1000>, f32, _> = m.forward(x.clone());
        println!("Forward time: {:?}", start.elapsed());
    }
}
