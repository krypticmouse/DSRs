use itertools::Itertools;

use crate::{
    ir_type::{TypeGeneric, UnionTypeGeneric},
    type_meta, ConstraintLevel,
};

impl TypeGeneric<type_meta::NonStreaming> {
    pub fn simplify(&self) -> Self {
        match self {
            TypeGeneric::Union(inner, union_meta) => {
                let view = inner.view();
                let flattened = view.flatten();
                let unique = flattened.into_iter().unique().collect::<Vec<_>>();
                let has_null = unique.contains(&TypeGeneric::null());
                // if the union contains null, we'll detect that here.
                let mut variants: Vec<TypeGeneric<type_meta::NonStreaming>> = unique
                    .into_iter()
                    .filter(|t| t != &TypeGeneric::null())
                    .collect::<Vec<_>>();

                // here metadata simplification of both variants and the union itself happens
                // unions will never have checks and asserts in their own metadata, always distributed and do not keep
                // Union(A|B)(@check(A, {..})) => Union(A@check(A, {..})|B@check(B, {..}))
                let (to_move, to_keep): (Vec<_>, Vec<_>) =
                    union_meta.constraints.clone().into_iter().partition(|c| {
                        // move these
                        matches!(c.level, ConstraintLevel::Check | ConstraintLevel::Assert)
                    });

                // Add to_move to each variant
                for variant in variants.iter_mut() {
                    variant.meta_mut().constraints.extend(to_move.clone());
                }

                let mut new_meta = type_meta::NonStreaming::default();
                new_meta.constraints.extend(to_keep);

                let simplified: TypeGeneric<type_meta::NonStreaming> = match variants.len() {
                    0 => return TypeGeneric::null(),
                    1 => {
                        if has_null {
                            // Return an optional of a single variant.
                            TypeGeneric::Union(
                                unsafe { UnionTypeGeneric::new_unsafe(vec![variants[0].clone()]) },
                                new_meta,
                            )
                        } else {
                            // Return the single variant.
                            variants[0].clone()
                        }
                    }
                    _ => {
                        if has_null {
                            variants.push(TypeGeneric::null());
                        }
                        TypeGeneric::Union(
                            unsafe { UnionTypeGeneric::new_unsafe(variants) },
                            new_meta,
                        )
                    }
                };

                simplified
            }
            _ => self.clone(),
        }
    }
}
