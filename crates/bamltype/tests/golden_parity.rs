use bamltype as facet_runtime;
use expect_test::expect;

/// Golden user docs.
#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "GoldenUser")]
#[baml(internal_name = "contract::golden::User")]
struct GoldenUser {
    /// Full name for display.
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "GoldenShape")]
#[baml(internal_name = "contract::golden::Shape")]
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
fn golden_schema_snapshot_user_default_fixture() {
    let schema =
        facet_runtime::render_schema::<GoldenUser>(facet_runtime::RenderOptions::default())
            .expect("facet render")
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
fn golden_schema_snapshot_shape_hoisted_fixture() {
    let schema = facet_runtime::render_schema::<GoldenShape>(
        facet_runtime::RenderOptions::hoist_classes(facet_runtime::HoistClasses::All),
    )
    .expect("facet render")
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
