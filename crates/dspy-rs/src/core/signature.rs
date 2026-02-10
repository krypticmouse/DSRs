use bamltype::Shape;
use facet::Facet;

use crate::{BamlType, OutputFormatContent};

use super::{FieldMetadataSpec, SignatureSchema};

#[derive(Debug, Clone, Copy)]
pub struct ConstraintSpec {
    pub kind: ConstraintKind,
    pub label: &'static str,
    pub expression: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    Check,
    Assert,
}

pub trait Signature: Send + Sync + 'static {
    type Input: BamlType + for<'a> Facet<'a> + Send + Sync;
    type Output: BamlType + for<'a> Facet<'a> + Send + Sync;

    fn instruction() -> &'static str;

    fn schema() -> &'static SignatureSchema
    where
        Self: Sized,
    {
        SignatureSchema::of::<Self>()
    }

    fn input_shape() -> &'static Shape;
    fn output_shape() -> &'static Shape;

    fn input_field_metadata() -> &'static [FieldMetadataSpec];
    fn output_field_metadata() -> &'static [FieldMetadataSpec];

    fn output_format_content() -> &'static OutputFormatContent
    where
        Self: Sized,
    {
        Self::schema().output_format()
    }
}
