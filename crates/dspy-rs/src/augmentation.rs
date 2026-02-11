use std::marker::PhantomData;
use std::ops::Deref;

use crate::{BamlType, Signature};
use facet::Facet;

/// Adds fields to a signature's output that the LM actually produces.
///
/// This is a prompt modification, not metadata. When [`ChainOfThought`](crate::ChainOfThought)
/// uses [`Reasoning`](crate::Reasoning), the LM literally sees `reasoning: String` in its
/// output format and generates text for it. Compare with [`CallMetadata`](crate::CallMetadata),
/// which is runtime bookkeeping the LM never sees.
///
/// Usually derived:
///
/// ```
/// use dspy_rs::*;
///
/// #[derive(Augmentation, Clone, Debug)]
/// #[augment(output, prepend)]
/// struct Confidence {
///     #[output] confidence: f64,
/// }
/// // Generates: WithConfidence<O> wrapper with Deref<Target = O>
/// ```
///
/// The generated wrapper implements `Deref<Target = O>`, so you get both the augmented
/// field (`result.confidence`) and the base fields (`result.answer`) without naming
/// the wrapper type.
///
/// Augmentations compose via tuples: `(Reasoning, Confidence)` wraps as
/// `WithReasoning<WithConfidence<O>>`. Auto-deref chains for field reads. Pattern
/// matching requires explicit destructuring through each layer — acceptable tradeoff.
pub trait Augmentation: Send + Sync + 'static {
    /// The wrapper type that adds this augmentation's fields around an inner output `T`.
    type Wrap<T: BamlType + for<'a> Facet<'a> + Send + Sync>: BamlType
        + for<'a> Facet<'a>
        + Deref
        + Send
        + Sync;
}

/// Type-level combinator: signature `S` with augmentation `A` applied to its output.
///
/// Same input as `S`, output is `A::Wrap<S::Output>`. This is how
/// [`ChainOfThought`](crate::ChainOfThought) works internally:
/// `Predict<Augmented<QA, Reasoning>>` has output `WithReasoning<QAOutput>`.
///
/// You typically don't use this directly — library modules wire it up for you.
/// Module authors use it when building new augmented strategies.
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

/// Convenience alias: the output type of `Augmented<S, A>`.
pub type AugmentedOutput<S, A> = <A as Augmentation>::Wrap<<S as Signature>::Output>;
