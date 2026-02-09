# S8 Spike: Facet Flatten Metadata Behavior

## Context

The entire SignatureSchema + adapter design hinges on detecting `#[facet(flatten)]` on struct fields and recursing into the inner type's fields. This was assumed but never verified against Facet's actual API.

## Goal

Determine exactly how `#[facet(flatten)]` manifests in Shape metadata and how to write a flatten-aware field walker for SignatureSchema derivation.

## Questions

| ID | Question |
|---|---|
| **S8-Q1** | When a struct has `#[facet(flatten)]` on a field, does the parent's StructType still show that field, or does Facet inline the inner fields? |
| **S8-Q2** | How do you detect a field is flattened? Attr? Flag? |
| **S8-Q3** | What does `field.shape()` return for a flattened field? |
| **S8-Q4** | Can we write a concrete walk that produces a flat field list from nested flatten? |

## Findings

1. **The field is still present in `StructType.fields`.** Facet does NOT inline inner fields at compile time. The derive macro sets `FieldFlags::FLATTEN` bit on the field (`facet-macros-impl/src/process_struct.rs:741`). The parent's `StructType.fields` slice has all declared fields, including the flattened one.

2. **Detection: `field.is_flattened()` — O(1) flag check.** `FieldFlags::FLATTEN = 1 << 1` (`facet-core/src/types/ty/field.rs:58`). Convenience method at `field.rs:169-175`:
   ```rust
   pub const fn is_flattened(&self) -> bool {
       self.flags.contains(FieldFlags::FLATTEN)
   }
   ```

3. **`field.shape()` returns the inner type's Shape.** For `#[facet(flatten)] inner: O`, `field.shape()` returns `O::SHAPE`. The flatten flag doesn't change what shape is stored — it's always the declared field type's shape.

4. **Facet already ships a flatten-aware iterator.** `fields_for_serialize()` in `facet-reflect/src/peek/fields.rs:452-626` uses a stack-based iterator that checks `is_flattened()` and recurses into inner struct fields. For schema-time (no values), the walk is:

   ```rust
   fn walk_fields_flat(shape: &'static Shape) -> Vec<&'static Field> {
       let Type::User(UserType::Struct(st)) = &shape.ty else { return vec![] };
       let mut result = Vec::new();
       for field in st.fields.iter() {
           if field.is_flattened() {
               result.extend(walk_fields_flat(field.shape()));
           } else {
               result.push(field);
           }
       }
       result
   }
   ```

   For `WithReasoning<QAOutput> { reasoning, #[flatten] inner: QAOutput { answer } }`, this yields `[reasoning, answer]`.

5. **Current bamltype code gaps:**
   - `schema_builder.rs` `build_struct_ir()`: iterates fields without checking `is_flattened()` — needs update
   - `convert.rs` deserialization (`build_object_fields`): no flatten awareness — needs update
   - `convert.rs` serialization: already works via Facet's `fields_for_serialize()`

## Decision

**Facet flatten works exactly as the design assumes.** The API is:
- `field.is_flattened()` to detect
- `field.shape()` to get inner type's Shape
- Recurse to produce flat field list

`SignatureSchema::derive()` will use this pattern in its `walk_fields` function. The design reference's pseudocode (`has_dsrs_flatten(field)`) maps directly to `field.is_flattened()`.

Known work when implementing:
- `schema_builder.rs` needs flatten-aware `build_struct_ir()`
- `convert.rs` deserialization needs flatten-aware `build_object_fields()`

## Acceptance

S8 is complete when:
- Facet's flatten API is documented with concrete code evidence
- The mapping from design pseudocode to real Facet API is established
- Gaps in bamltype code are identified
