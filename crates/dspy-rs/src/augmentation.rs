use std::marker::PhantomData;
use std::ops::Deref;

use crate::{BamlType, Signature};
use facet::Facet;

pub trait Augmentation: Send + Sync + 'static {
    type Wrap<T: BamlType + for<'a> Facet<'a> + Send + Sync>: BamlType
        + for<'a> Facet<'a>
        + Deref
        + Send
        + Sync;
}

#[derive(Clone, Copy, Default)]
pub struct Augmented<S: Signature, A: Augmentation> {
    _marker: PhantomData<(S, A)>,
}

impl<S: Signature, A: Augmentation> Signature for Augmented<S, A> {
    type Input = S::Input;
    type Output = A::Wrap<S::Output>;

    fn instruction() -> &'static str {
        S::instruction()
    }

    fn input_shape() -> &'static bamltype::Shape {
        S::input_shape()
    }

    fn output_shape() -> &'static bamltype::Shape {
        <A::Wrap<S::Output> as Facet<'static>>::SHAPE
    }

    fn input_field_metadata() -> &'static [crate::FieldMetadataSpec] {
        S::input_field_metadata()
    }

    fn output_field_metadata() -> &'static [crate::FieldMetadataSpec] {
        S::output_field_metadata()
    }
}

impl<A: Augmentation, B: Augmentation> Augmentation for (A, B) {
    type Wrap<T: BamlType + for<'a> Facet<'a> + Send + Sync> = A::Wrap<B::Wrap<T>>;
}

pub type AugmentedOutput<S, A> = <A as Augmentation>::Wrap<<S as Signature>::Output>;
