use crate::{BamlType, CallOutcome, CallOutcomeErrorKind, Facet};

use super::Module;

pub trait ModuleExt: Module + Sized {
    fn map<F, T>(self, map: F) -> Map<Self, F>
    where
        F: Fn(Self::Output) -> T + Send + Sync + 'static,
        T: BamlType + for<'a> Facet<'a> + Send + Sync,
    {
        Map {
            inner: self,
            map: facet::Opaque(map),
        }
    }

    fn and_then<F, T>(self, and_then: F) -> AndThen<Self, F>
    where
        F: Fn(Self::Output) -> Result<T, CallOutcomeErrorKind> + Send + Sync + 'static,
        T: BamlType + for<'a> Facet<'a> + Send + Sync,
    {
        AndThen {
            inner: self,
            and_then: facet::Opaque(and_then),
        }
    }
}

impl<M: Module> ModuleExt for M {}

pub struct Map<M, F: 'static> {
    pub(crate) inner: M,
    map: facet::Opaque<F>,
}

unsafe fn map_drop<M, F: 'static>(ox: facet::OxPtrMut) {
    unsafe {
        core::ptr::drop_in_place(ox.ptr().as_byte_ptr() as *mut Map<M, F>);
    }
}

// `derive(Facet)` currently imposes `F: Facet` for these generic wrappers.
// We intentionally model closure fields as skipped opaque data and only expose `inner`.
unsafe impl<'a, M, F> facet::Facet<'a> for Map<M, F>
where
    M: facet::Facet<'a>,
    F: 'static,
{
    const SHAPE: &'static facet::Shape = &const {
        const fn build_type_ops<M, F: 'static>() -> facet::TypeOpsIndirect {
            facet::TypeOpsIndirect {
                drop_in_place: map_drop::<M, F>,
                default_in_place: None,
                clone_into: None,
                is_truthy: None,
            }
        }

        facet::ShapeBuilder::for_sized::<Map<M, F>>("Map")
            .module_path(module_path!())
            .ty(facet::Type::User(facet::UserType::Struct(facet::StructType {
                repr: facet::Repr::default(),
                kind: facet::StructKind::Struct,
                fields: &const {
                    [
                        facet::FieldBuilder::new(
                            "inner",
                            facet::shape_of::<M>,
                            core::mem::offset_of!(Map<M, F>, inner),
                        )
                        .build(),
                        facet::FieldBuilder::new(
                            "map",
                            facet::shape_of::<facet::Opaque<F>>,
                            core::mem::offset_of!(Map<M, F>, map),
                        )
                        .flags(facet::FieldFlags::SKIP)
                        .build(),
                    ]
                },
            })))
            .def(facet::Def::Scalar)
            .type_params(&[
                facet::TypeParam {
                    name: "M",
                    shape: M::SHAPE,
                },
                facet::TypeParam {
                    name: "F",
                    shape: <facet::Opaque<F> as facet::Facet<'a>>::SHAPE,
                },
            ])
            .vtable_indirect(&facet::VTableIndirect::EMPTY)
            .type_ops_indirect(&const { build_type_ops::<M, F>() })
            .build()
    };
}

#[allow(async_fn_in_trait)]
impl<M, F, T> Module for Map<M, F>
where
    M: Module,
    F: Fn(M::Output) -> T + Send + Sync + 'static,
    T: BamlType + for<'a> Facet<'a> + Send + Sync,
{
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let (result, metadata) = self.inner.forward(input).await.into_parts();
        match result {
            Ok(output) => CallOutcome::ok((self.map.0)(output), metadata),
            Err(err) => CallOutcome::err(err, metadata),
        }
    }
}

pub struct AndThen<M, F: 'static> {
    pub(crate) inner: M,
    and_then: facet::Opaque<F>,
}

unsafe fn and_then_drop<M, F: 'static>(ox: facet::OxPtrMut) {
    unsafe {
        core::ptr::drop_in_place(ox.ptr().as_byte_ptr() as *mut AndThen<M, F>);
    }
}

// See `Map` above: closure type `F` is intentionally opaque and skipped.
unsafe impl<'a, M, F> facet::Facet<'a> for AndThen<M, F>
where
    M: facet::Facet<'a>,
    F: 'static,
{
    const SHAPE: &'static facet::Shape = &const {
        const fn build_type_ops<M, F: 'static>() -> facet::TypeOpsIndirect {
            facet::TypeOpsIndirect {
                drop_in_place: and_then_drop::<M, F>,
                default_in_place: None,
                clone_into: None,
                is_truthy: None,
            }
        }

        facet::ShapeBuilder::for_sized::<AndThen<M, F>>("AndThen")
            .module_path(module_path!())
            .ty(facet::Type::User(facet::UserType::Struct(facet::StructType {
                repr: facet::Repr::default(),
                kind: facet::StructKind::Struct,
                fields: &const {
                    [
                        facet::FieldBuilder::new(
                            "inner",
                            facet::shape_of::<M>,
                            core::mem::offset_of!(AndThen<M, F>, inner),
                        )
                        .build(),
                        facet::FieldBuilder::new(
                            "and_then",
                            facet::shape_of::<facet::Opaque<F>>,
                            core::mem::offset_of!(AndThen<M, F>, and_then),
                        )
                        .flags(facet::FieldFlags::SKIP)
                        .build(),
                    ]
                },
            })))
            .def(facet::Def::Scalar)
            .type_params(&[
                facet::TypeParam {
                    name: "M",
                    shape: M::SHAPE,
                },
                facet::TypeParam {
                    name: "F",
                    shape: <facet::Opaque<F> as facet::Facet<'a>>::SHAPE,
                },
            ])
            .vtable_indirect(&facet::VTableIndirect::EMPTY)
            .type_ops_indirect(&const { build_type_ops::<M, F>() })
            .build()
    };
}

#[allow(async_fn_in_trait)]
impl<M, F, T> Module for AndThen<M, F>
where
    M: Module,
    F: Fn(M::Output) -> Result<T, CallOutcomeErrorKind> + Send + Sync + 'static,
    T: BamlType + for<'a> Facet<'a> + Send + Sync,
{
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let (result, metadata) = self.inner.forward(input).await.into_parts();
        match result {
            Ok(output) => match (self.and_then.0)(output) {
                Ok(transformed) => CallOutcome::ok(transformed, metadata),
                Err(err) => CallOutcome::err(err, metadata),
            },
            Err(err) => CallOutcome::err(err, metadata),
        }
    }
}
