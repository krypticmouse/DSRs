use baml_bridge::{render_schema, BamlType, HoistClasses, RenderOptions};
use expect_test::expect;

/// Golden user docs.
#[derive(Debug, Clone, PartialEq, BamlType)]
struct GoldenUser {
    /// Full name for display.
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
#[baml(tag = "kind")]
enum GoldenShape {
    /// A circle.
    Circle {
        radius: f64,
    },
    Square {
        side: f64,
    },
}

#[test]
fn schema_snapshot_user_default() {
    let schema = render_schema::<GoldenUser>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    expect![[r#"Answer in JSON using this schema:
{
  // Golden user docs.

  // Full name for display.
  fullName: string,
  age: int,
}"#]]
    .assert_eq(&schema);
}

#[test]
fn schema_snapshot_shape_hoisted() {
    let schema = render_schema::<GoldenShape>(RenderOptions::hoist_classes(HoistClasses::All))
        .expect("render failed")
        .unwrap_or_default();

    expect![[r#"GoldenShape_Circle {
  // A circle.

  kind: "Circle",
  radius: float,
}

GoldenShape_Square {
  kind: "Square",
  side: float,
}

Answer in JSON using any of these schemas:
GoldenShape_Circle or GoldenShape_Square"#]]
    .assert_eq(&schema);
}
