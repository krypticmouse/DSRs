use dspy_rs::ir::compiler::{Compiler, CompilerMeasurement, fusion_next, transformation_menu};
use dspy_rs::ir::{
    self, Edit, FieldType as T, Interpreter, Node, Program, ProgramBuilder, RuntimeEnv,
    SignatureDef,
};
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
use rig::completion::AssistantContent;
use std::sync::Arc;

fn program() -> Program {
    let mut b = ProgramBuilder::new("independent");
    b.model("m", LMConfig::default());
    let main = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .output("evidence", T::String)
            .finish()
            .unwrap(),
    );
    let answer = b.sig(
        SignatureDef::build("Answer")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let evidence = b.sig(
        SignatureDef::build("Evidence")
            .instruction("Give supporting evidence.")
            .input("question", T::String)
            .output("evidence", T::String)
            .finish()
            .unwrap(),
    );
    b.main(
        main,
        ir::seq([
            ir::predict("answerer", answer).bind("question", ir::input("question")),
            ir::predict("researcher", evidence).bind("question", ir::input("question")),
        ])
        .out("answer", ir::out("answerer", "answer"))
        .out("evidence", ir::out("researcher", "evidence")),
    )
    .unwrap()
}

fn fused(p: &Program) -> Program {
    p.edited(&[Edit::Fuse {
        first: p.leaf_id("answerer").unwrap(),
        second: p.leaf_id("researcher").unwrap(),
    }])
    .unwrap()
}

fn round_trip(p: &Program) {
    p.validate().unwrap();
    let text = p.to_dsrs();
    assert_eq!(
        Program::from_dsrs(&text).unwrap().meta.program_hash,
        p.meta.program_hash
    );
    let loaded: Program = serde_json::from_str(&serde_json::to_string(p).unwrap()).unwrap();
    assert_eq!(loaded.meta.program_hash, p.meta.program_hash);
}

#[test]
fn fusion_rewrites_exports_and_round_trips() {
    let p = program();
    let result = fused(&p);
    assert_eq!(
        p.nodes
            .values()
            .filter(|n| matches!(n, Node::Predict(_)))
            .count(),
        2
    );
    assert_eq!(
        result
            .nodes
            .values()
            .filter(|n| matches!(n, Node::Predict(_)))
            .count(),
        1
    );
    assert!(result.leaf_id("researcher").is_none());
    round_trip(&result);
}

#[test]
fn fusion_rejects_cross_collisions_between_output_names_and_aliases() {
    let mut p = program();
    let first = p.leaf_id("answerer").unwrap();
    let second = p.leaf_id("researcher").unwrap();
    let Node::Predict(a) = p.nodes[first].clone() else {
        unreachable!()
    };
    let Node::Predict(b) = p.nodes[second].clone() else {
        unreachable!()
    };
    p.sigs[a.sig].outputs[0].lm_name = "response".into();
    p.sigs[b.sig].outputs[0].lm_name = "answer".into();
    p.validate().unwrap();
    assert!(fusion_next(&p, first).is_none());
    assert!(p.edited(&[Edit::Fuse { first, second }]).is_err());
}

#[test]
fn split_preserves_all_outputs_and_round_trips() {
    let p = fused(&program());
    let result = p
        .edited(&[Edit::Split {
            leaf: p.leaf_id("answerer").unwrap(),
            outputs: vec!["evidence".into()],
            name: "evidence_step".into(),
        }])
        .unwrap();
    assert_eq!(
        result
            .nodes
            .values()
            .filter(|n| matches!(n, Node::Predict(_)))
            .count(),
        2
    );
    assert!(result.leaf_id("evidence_step").is_some());
    round_trip(&result);
}

#[test]
fn invalid_splits_cannot_mutate_the_parent() {
    let p = fused(&program());
    let hash = p.meta.program_hash;
    for outputs in [
        vec![],
        vec!["missing".into()],
        vec!["answer".into(), "evidence".into()],
        vec!["answer".into(), "answer".into()],
    ] {
        assert!(
            p.edited(&[Edit::Split {
                leaf: p.leaf_id("answerer").unwrap(),
                outputs,
                name: "child".into()
            }])
            .is_err()
        );
    }
    assert_eq!(p.meta.program_hash, hash);
}

#[test]
fn fusion_rejects_different_models_and_input_bindings() {
    let mut p = program();
    let first = p.leaf_id("answerer").unwrap();
    let second = p.leaf_id("researcher").unwrap();
    let Node::Predict(n) = &mut p.nodes[second] else {
        unreachable!()
    };
    n.binding[0].src = ir::PortRef::Lit("different question".into());
    assert!(fusion_next(&p, first).is_none());
    assert!(p.edited(&[Edit::Fuse { first, second }]).is_err());
}

async fn execute(
    p: Program,
    replies: Vec<&str>,
) -> (serde_json::Map<String, serde_json::Value>, usize) {
    let provider = TestCompletionModel::new(replies.into_iter().map(AssistantContent::text));
    let lm = LM::builder()
        .base_url("http://127.0.0.1:1/v1".into())
        .model("compiler".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(provider.clone()))
        .await
        .unwrap();
    let interpreter = Interpreter::load(p, RuntimeEnv::new().bind_model("m", Arc::new(lm)))
        .await
        .unwrap();
    let output = interpreter
        .run(
            [("question".into(), "France?".into())]
                .into_iter()
                .collect(),
            None,
            ir::Budget::unlimited(),
        )
        .await
        .unwrap();
    (output, provider.call_count())
}

#[tokio::test]
async fn fused_and_split_programs_execute_with_equivalent_outputs() {
    let p = program();
    let fused = fused(&p);
    let split = fused
        .edited(&[Edit::Split {
            leaf: fused.leaf_id("answerer").unwrap(),
            outputs: vec!["evidence".into()],
            name: "evidence_step".into(),
        }])
        .unwrap();
    let a = "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]";
    let b = "[[ ## evidence ## ]]\nParis is the capital of France.\n[[ ## completed ## ]]";
    let both = "[[ ## answer ## ]]\nParis\n[[ ## evidence ## ]]\nParis is the capital of France.\n[[ ## completed ## ]]";
    let (before, before_calls) = execute(p, vec![a, b]).await;
    let (after, after_calls) = execute(fused, vec![both]).await;
    let (split, split_calls) = execute(split, vec![a, b]).await;
    assert_eq!(before, after);
    assert_eq!(before, split);
    assert_eq!((before_calls, after_calls, split_calls), (2, 1, 2));
}

fn measurement(quality: f64, calls: u64) -> CompilerMeasurement {
    CompilerMeasurement {
        quality: vec![quality, quality],
        latency_us: vec![calls * 10_000, calls * 12_000],
        lm_calls: calls * 2,
        feedback: vec!["Answers and evidence checked against the same fixtures".into()],
    }
}

#[tokio::test]
async fn compiler_promotes_faster_candidates_without_quality_loss() {
    let p = program();
    let report = Compiler::default()
        .compile(&p, |p| async move {
            let calls = p
                .nodes
                .values()
                .filter(|n| matches!(n, Node::Predict(_)))
                .count() as u64;
            Ok(measurement(1.0, calls))
        })
        .await
        .unwrap();
    assert_eq!(
        report.final_measurement.lm_calls,
        report.baseline.lm_calls / 2
    );
    assert!(report.steps.iter().any(|step| step.accepted));
    assert!(report.steps.iter().any(|step| !step.accepted));
    round_trip(&report.program);
}

#[tokio::test]
async fn compiler_rejects_quality_regressions_and_invalid_measurements() {
    for invalid in [false, true] {
        let p = program();
        let base = p.meta.program_hash;
        let report = Compiler::default()
            .compile(&p, |p| async move {
                if p.meta.program_hash == base {
                    Ok(measurement(1.0, 2))
                } else if invalid {
                    let mut m = measurement(1.0, 1);
                    m.quality[0] = f64::NAN;
                    Ok(m)
                } else {
                    Ok(measurement(0.5, 1))
                }
            })
            .await
            .unwrap();
        assert_eq!(report.program.meta.program_hash, base);
        assert!(report.steps.iter().all(|step| !step.accepted));
        let encoded = serde_json::to_string(&report).unwrap();
        let _: ir::compiler::CompilerReport = serde_json::from_str(&encoded).unwrap();
    }
}

#[test]
fn transformations_are_available_to_feedback_driven_selection() {
    let p = program();
    assert!(
        p.legal_edits(p.leaf_id("answerer").unwrap())
            .contains(&ir::EditKind::FuseNext)
    );
    let fused = fused(&p);
    assert!(
        fused
            .legal_edits(fused.leaf_id("answerer").unwrap())
            .contains(&ir::EditKind::SplitOutputs)
    );
    assert!(!transformation_menu(&p).is_empty());
}

#[test]
fn splitting_a_single_prediction_program_preserves_its_public_interface() {
    let mut b = ProgramBuilder::new("root_split");
    b.model("m", LMConfig::default());
    let sig = b.sig(
        SignatureDef::build("Pair")
            .input("question", T::String)
            .output("a", T::String)
            .output("b", T::String)
            .finish()
            .unwrap(),
    );
    let p = b
        .main(
            sig,
            ir::seq([ir::predict("pair", sig).bind("question", ir::input("question"))])
                .out("a", ir::out("pair", "a"))
                .out("b", ir::out("pair", "b")),
        )
        .unwrap();
    let result = p
        .edited(&[Edit::Split {
            leaf: p.leaf_id("pair").unwrap(),
            outputs: vec!["b".into()],
            name: "second".into(),
        }])
        .unwrap();
    assert!(matches!(result.nodes[result.root], Node::Seq(_)));
    assert_eq!(result.sigs[result.sig], p.sigs[p.sig]);
    round_trip(&result);
    assert!(
        p.edited(&[Edit::Split {
            leaf: p.leaf_id("pair").unwrap(),
            outputs: vec!["b".into()],
            name: "invalid name".into()
        }])
        .is_err()
    );
}

#[test]
fn fusion_rejects_demos_and_model_changes() {
    let mut p = program();
    let first = p.leaf_id("answerer").unwrap();
    let second = p.leaf_id("researcher").unwrap();
    let Node::Predict(n) = &p.nodes[second] else {
        unreachable!()
    };
    let demos = n.demos;
    p.params[demos].default = ir::ParamValue::Demos {
        rows: vec![ir::DemoRow {
            input: [("question".into(), "?".into())].into_iter().collect(),
            output: [("evidence".into(), "fact".into())].into_iter().collect(),
        }],
    };
    assert!(fusion_next(&p, first).is_none());
    p.params[demos].default = ir::ParamValue::Demos { rows: Vec::new() };
    let Node::Predict(n) = &p.nodes[second] else {
        unreachable!()
    };
    let model = n.model;
    let model_id = p.models.push(ir::ModelDef {
        name: "different".into(),
        config: LMConfig {
            model: "different".into(),
            ..LMConfig::default()
        },
    });
    p.params[model].default = ir::ParamValue::ModelRef { model: model_id };
    assert!(fusion_next(&p, first).is_none());
}

#[tokio::test]
async fn evaluation_budget_includes_the_baseline() {
    let compiler = Compiler {
        max_evaluations: 1,
        ..Compiler::default()
    };
    let report = compiler
        .compile(&program(), |_| async { Ok(measurement(1.0, 2)) })
        .await
        .unwrap();
    assert!(report.steps.is_empty());
    assert_eq!(
        report.program.meta.program_hash,
        program().meta.program_hash
    );
}
