use rstest::*;
use dsrs::signature::signature::Signature;
use dsrs::signature::field::Field;

#[rstest]
fn test_field_initizalization() {
    let input_field = Field::InputField {
        prefix: "prefix".to_string(),
        desc: "desc".to_string(),
        format: None,
        output_type: "output_type".to_string(),
    };

    let output_field = Field::OutputField {
        prefix: "prefix".to_string(),
        desc: "desc".to_string(),
        format: None,
        output_type: "output_type".to_string(),
    };
    assert_eq!(input_field.prefix(), "prefix");
    assert_eq!(input_field.desc(), "desc");
    assert_eq!(input_field.format(), None);
    assert_eq!(input_field.output_type(), "output_type");

    assert_eq!(output_field.prefix(), "prefix");
    assert_eq!(output_field.desc(), "desc");
    assert_eq!(output_field.format(), None);
    assert_eq!(output_field.output_type(), "output_type");
}

#[rstest]
fn test_signature_from_string() {
    let signature = Signature::from("inp1, inp2 -> out1, out2".to_string());

    assert_eq!(signature.description, "Given a inputs inp1, inp2, return outputs out1, out2");
    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);
}

#[rstest]
fn test_signature_insert() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2".to_string());
    signature.insert("inp3".to_string(), Field::InputField {
        prefix: "prefix".to_string(),
        desc: "desc".to_string(),
        format: None,
        output_type: "output_type".to_string(),
    }, 0);

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().prefix(), "prefix");
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp3").unwrap().format(), None);
    assert_eq!(signature.input_fields.get("inp3").unwrap().output_type(), "output_type");

    assert_eq!(signature.input_fields.get("inp1").unwrap().prefix(), "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().format(), None);
    assert_eq!(signature.input_fields.get("inp1").unwrap().output_type(), "");
}

#[rstest]
fn test_signature_append() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2".to_string());
    signature.append("inp3".to_string(), Field::InputField {
        prefix: "prefix".to_string(),
        desc: "desc".to_string(),
        format: None,
        output_type: "output_type".to_string(),
    });

    assert_eq!(signature.input_fields.len(), 3);
    
    assert_eq!(signature.input_fields.get("inp3").unwrap().prefix(), "prefix");
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp3").unwrap().format(), None);
    assert_eq!(signature.input_fields.get("inp3").unwrap().output_type(), "output_type");

    assert_eq!(signature.input_fields.get("inp1").unwrap().prefix(), "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().format(), None);
    assert_eq!(signature.input_fields.get("inp1").unwrap().output_type(), "");
}

#[rstest]
fn test_signature_prepend() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2".to_string());
    signature.prepend("inp3".to_string(), Field::InputField {
        prefix: "prefix".to_string(),
        desc: "desc".to_string(),
        format: None,
        output_type: "output_type".to_string(),
    });

    assert_eq!(signature.input_fields.len(), 3);
    
    assert_eq!(signature.input_fields.get("inp3").unwrap().prefix(), "prefix");
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp3").unwrap().format(), None);
    assert_eq!(signature.input_fields.get("inp3").unwrap().output_type(), "output_type");

    assert_eq!(signature.input_fields.get("inp1").unwrap().prefix(), "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().format(), None);
    assert_eq!(signature.input_fields.get("inp1").unwrap().output_type(), "");
}
