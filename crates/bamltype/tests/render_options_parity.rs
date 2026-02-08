use std::collections::HashMap;

use bamltype as facet_runtime;
use expect_test::expect;

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "RenderLeaf")]
#[baml(internal_name = "contract::render::Leaf")]
struct RenderLeaf {
    value: String,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "RenderChoice")]
#[baml(internal_name = "contract::render::Choice")]
#[baml(tag = "kind")]
enum RenderChoice {
    First { leaf: RenderLeaf },
    Second { score: i64 },
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "RenderRoot")]
#[baml(internal_name = "contract::render::Root")]
struct RenderRoot {
    metadata: HashMap<String, String>,
    choice: RenderChoice,
}

fn render(options: facet_runtime::RenderOptions) -> String {
    facet_runtime::render_schema::<RenderRoot>(options)
        .expect("facet render")
        .unwrap_or_default()
}

#[test]
fn render_options_default_fixture() {
    let rendered = render(facet_runtime::RenderOptions::default());
    expect![[r#"
        Answer in JSON using this schema:
        {
          metadata: map<string, string>,
          choice: {
            kind: "First",
            leaf: {
              value: string,
            },
          } or {
            kind: "Second",
            score: int,
          },
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn render_options_map_style_object_literal_fixture() {
    let rendered = render(
        facet_runtime::RenderOptions::default()
            .with_map_style(facet_runtime::MapStyle::ObjectLiteral),
    );
    expect![[r#"
        Answer in JSON using this schema:
        {
          metadata: {string: string},
          choice: {
            kind: "First",
            leaf: {
              value: string,
            },
          } or {
            kind: "Second",
            score: int,
          },
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn render_options_quote_class_fields_fixture() {
    let rendered = render(facet_runtime::RenderOptions::default().with_quote_class_fields(true));
    expect![[r#"
        Answer in JSON using this schema:
        {
          "metadata": map<string, string>,
          "choice": {
            "kind": "First",
            "leaf": {
              "value": string,
            },
          } or {
            "kind": "Second",
            "score": int,
          },
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn render_options_prefix_fixture() {
    let rendered = render(
        facet_runtime::RenderOptions::default().with_prefix(Some("contract schema".to_string())),
    );
    expect![[r#"
        contract schema{
          metadata: map<string, string>,
          choice: {
            kind: "First",
            leaf: {
              value: string,
            },
          } or {
            kind: "Second",
            score: int,
          },
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn render_options_or_splitter_fixture() {
    let rendered = render(
        facet_runtime::RenderOptions::hoist_classes(facet_runtime::HoistClasses::All)
            .with_or_splitter(" || "),
    );
    expect![[r#"
        RenderChoice_First {
          kind: "First",
          leaf: RenderLeaf,
        }

        RenderChoice_Second {
          kind: "Second",
          score: int,
        }

        RenderLeaf {
          value: string,
        }

        RenderRoot {
          metadata: map<string, string>,
          choice: RenderChoice_First || RenderChoice_Second,
        }

        Answer in JSON using this schema: RenderRoot"#]]
    .assert_eq(&rendered);
}

#[test]
fn render_options_hoist_subset_fixture() {
    let rendered = render(facet_runtime::RenderOptions::default().with_hoist_classes(
        facet_runtime::HoistClasses::Subset(vec!["contract::render::Leaf".to_string()]),
    ));
    expect![[r#"
        RenderLeaf {
          value: string,
        }

        Answer in JSON using this schema:
        {
          metadata: map<string, string>,
          choice: {
            kind: "First",
            leaf: RenderLeaf,
          } or {
            kind: "Second",
            score: int,
          },
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn render_options_new_constructor_fixture() {
    let rendered = render(facet_runtime::RenderOptions::new(
        Some(Some("contract schema".to_string())),
        Some(" || ".to_string()),
        Some(Some("* ".to_string())),
        Some(true),
        Some(facet_runtime::MapStyle::ObjectLiteral),
        Some(Some("TYPE ".to_string())),
        Some(facet_runtime::HoistClasses::All),
        Some(true),
    ));

    expect![[r#"
        TYPE  RenderChoice_First {
          "kind": "First",
          "leaf": RenderLeaf,
        }

        TYPE  RenderChoice_Second {
          "kind": "Second",
          "score": int,
        }

        TYPE  RenderLeaf {
          "value": string,
        }

        TYPE  RenderRoot {
          "metadata": {string: string},
          "choice": RenderChoice_First || RenderChoice_Second,
        }

        contract schemaRenderRoot"#]]
    .assert_eq(&rendered);
}
