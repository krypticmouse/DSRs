//! The graph-edit calculus (`ir::edit`): typed structural edits over the IR —
//! application, apply/validate failure modes, garbage collection, the
//! `legal_edits` menu, and overlay migration across structural change.

use std::num::NonZeroU32;

use dspy_rs::LMConfig;
use dspy_rs::ir::{
    self, ApplyError, CapSet, DemoRow, Edit, EditError, EditKind, FieldDef, FieldType as T, Node,
    NodeBudget, NodeId, Overlay, ParamKind, ParamValue, PortRef, Program, ProgramBuilder,
    SignatureDef, StopSpec, SwapTarget, ToolId, ValidateError, migrate_overlay,
};
use serde_json::json;

fn model_config(name: &str) -> LMConfig {
    LMConfig {
        model: name.to_string(),
        ..LMConfig::default()
    }
}

fn obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// A stale/fabricated NodeId (ids are serde-transparent u32s).
fn nid(raw: u32) -> NodeId {
    serde_json::from_value(json!(raw)).unwrap()
}

fn tid(raw: u32) -> ToolId {
    serde_json::from_value(json!(raw)).unwrap()
}

fn tool_id(p: &Program, name: &str) -> ToolId {
    p.tools
        .iter()
        .find_map(|(id, tool)| (p.syms.get(tool.name) == name).then_some(id))
        .unwrap_or_else(|| panic!("tool `{name}` exists"))
}

/// The exact `cot` reasoning field the builder's `cot()` sugar prepends.
fn reasoning_field() -> FieldDef {
    FieldDef::new("reasoning", T::String).with_docs("Think step by step to reach the answer.")
}

/// Full round trip: canonical text parses back to the same hash and re-prints
/// byte-identically; the JSON projection loads to the same hash.
fn assert_round_trips(p: &Program) {
    let text = p.to_dsrs();
    let reparsed = Program::from_dsrs(&text).expect("canonical text parses");
    assert_eq!(reparsed.meta.program_hash, p.meta.program_hash);
    assert_eq!(reparsed.to_dsrs(), text, "print(parse(t)) == t");

    let json = serde_json::to_string(p).unwrap();
    let loaded: Program = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.meta.program_hash, p.meta.program_hash);
}

/// question → drafter (QA) → checker (Check) → verdict.
fn pipeline() -> Program {
    let mut b = ProgramBuilder::new("pipeline");
    b.model("m", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let check = b.sig(
        SignatureDef::build("Check")
            .instruction("Judge the answer.")
            .input("answer", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let checker = ir::predict("checker", check).bind("answer", ir::out("drafter", "answer"));
    b.main(
        main_sig,
        ir::seq([drafter, checker]).out("verdict", ir::out("checker", "verdict")),
    )
    .unwrap()
}

/// question → researcher (agent, tools [search], stop_tools [search]) →
/// summarizer (predict) → summary. Declares a second tool (calc) nothing uses.
fn agent_program() -> Program {
    let mut b = ProgramBuilder::new("agents");
    b.cap("net:search");
    b.cap("math:eval");
    b.model("m", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("summary", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .instruction("Research the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let sum = b.sig(
        SignatureDef::build("Sum")
            .instruction("Summarize the answer.")
            .input("answer", T::String)
            .output("summary", T::String)
            .finish()
            .unwrap(),
    );
    let search_sig = b.sig(
        SignatureDef::build("Search")
            .input("query", T::String)
            .output("results", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let calc_sig = b.sig(
        SignatureDef::build("Calc")
            .input("expr", T::String)
            .output("value", T::String)
            .finish()
            .unwrap(),
    );
    let search = b.host_tool("search", "Web search", search_sig, &["net:search"]);
    let _calc = b.host_tool("calc", "Calculator", calc_sig, &["math:eval"]);
    let researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .tools([search])
        .stop_tools([search])
        .max_turns(6);
    let summarizer = ir::predict("summarizer", sum).bind("answer", ir::out("researcher", "answer"));
    b.main(
        main_sig,
        ir::seq([researcher, summarizer]).out("summary", ir::out("summarizer", "summary")),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Edits are serde values
// ---------------------------------------------------------------------------

#[test]
fn every_edit_variant_serde_round_trips() {
    let edits = vec![
        Edit::AugmentSig {
            leaf: nid(1),
            prepend: reasoning_field(),
        },
        Edit::SwapLeaf {
            leaf: nid(1),
            to: SwapTarget::Agent {
                tools: vec![tid(0)],
                stop: StopSpec::default(),
                budget: NodeBudget::default(),
            },
        },
        Edit::SwapLeaf {
            leaf: nid(2),
            to: SwapTarget::Predict,
        },
        Edit::WrapRetry {
            node: nid(3),
            max_attempts: NonZeroU32::new(2).unwrap(),
            backoff_ms: 250,
            feedback: true,
        },
        Edit::Remove { node: nid(4) },
        Edit::AddTool {
            agent: nid(1),
            tool: tid(1),
        },
        Edit::RemoveTool {
            agent: nid(1),
            tool: tid(0),
        },
        Edit::SetStop {
            agent: nid(1),
            stop: StopSpec {
                max_turns: NonZeroU32::new(3).unwrap(),
                stop_tools: Box::new([tid(0)]),
                until_parse: false,
            },
        },
        Edit::SetInstructionDefault {
            leaf: nid(1),
            text: "Be terse.".to_string(),
        },
    ];
    for edit in &edits {
        let json = serde_json::to_string(edit).unwrap();
        let back: Edit = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, edit, "round trip of {json}");
    }
    // The whole list round-trips as one value (edits are replayable scripts).
    let json = serde_json::to_string(&edits).unwrap();
    let back: Vec<Edit> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, edits);

    // EditKind is serde too (the proposer menu is promptable data).
    let kinds = vec![
        EditKind::AugmentSig,
        EditKind::AddTool { tool: tid(1) },
        EditKind::SetStop,
    ];
    let json = serde_json::to_string(&kinds).unwrap();
    let back: Vec<EditKind> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, kinds);
}

// ---------------------------------------------------------------------------
// AugmentSig
// ---------------------------------------------------------------------------

#[test]
fn augment_sig_is_the_cot_move() {
    let parent = pipeline();
    let drafter = parent.leaf_id("drafter").unwrap();
    let old_sig = match &parent.nodes[drafter] {
        Node::Predict(n) => n.sig,
        _ => unreachable!(),
    };

    let child = parent
        .edited(&[Edit::AugmentSig {
            leaf: drafter,
            prepend: reasoning_field(),
        }])
        .unwrap();

    // New SigId, reasoning prepended, base outputs preserved.
    let new_sig = match &child.nodes[child.leaf_id("drafter").unwrap()] {
        Node::Predict(n) => n.sig,
        _ => unreachable!(),
    };
    assert_ne!(new_sig, old_sig);
    let sig = &child.sigs[new_sig];
    assert_eq!(&*sig.outputs[0].name, "reasoning");
    assert_eq!(&*sig.outputs[1].name, "answer");
    assert_eq!(&*sig.name, "QA", "the cot move keeps the base name");

    // The parent is untouched; the child re-validates under a new hash.
    assert_eq!(parent.sigs[old_sig].outputs.len(), 1);
    assert_ne!(child.meta.program_hash, parent.meta.program_hash);
    child.validate().unwrap();

    // The canonical text re-sugars the augmented Predict.
    assert!(child.to_dsrs().contains("cot QA"));
    assert_round_trips(&child);
}

#[test]
fn augment_sig_with_other_fields_renames_the_new_signature() {
    // Two leaves share one SigId: copy-on-write must not disturb the sibling.
    let mut b = ProgramBuilder::new("shared");
    b.model("m", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let first = ir::predict("first", qa).bind("question", ir::input("question"));
    let second = ir::predict("second", qa).bind("question", ir::out("first", "answer"));
    let parent = b
        .main(
            main_sig,
            ir::seq([first, second]).out("answer", ir::out("second", "answer")),
        )
        .unwrap();

    let first_id = parent.leaf_id("first").unwrap();
    let child = parent
        .edited(&[Edit::AugmentSig {
            leaf: first_id,
            prepend: FieldDef::new("plan", T::String),
        }])
        .unwrap();

    let (first_sig, second_sig) = (
        match &child.nodes[child.leaf_id("first").unwrap()] {
            Node::Predict(n) => n.sig,
            _ => unreachable!(),
        },
        match &child.nodes[child.leaf_id("second").unwrap()] {
            Node::Predict(n) => n.sig,
            _ => unreachable!(),
        },
    );
    // The sibling keeps the shared signature untouched.
    assert_ne!(first_sig, second_sig);
    assert_eq!(&*child.sigs[second_sig].name, "QA");
    assert_eq!(child.sigs[second_sig].outputs.len(), 1);
    // The augmented copy got a fresh name (two `sig QA` blocks cannot print).
    assert_eq!(&*child.sigs[first_sig].name, "QA_plan");
    assert_eq!(&*child.sigs[first_sig].outputs[0].name, "plan");
    assert!(child.to_dsrs().contains("sig QA_plan"));
    assert_round_trips(&child);
}

#[test]
fn augment_sig_rejects_duplicate_fields_and_containers() {
    let parent = pipeline();
    let drafter = parent.leaf_id("drafter").unwrap();

    let err = parent
        .edited(&[Edit::AugmentSig {
            leaf: drafter,
            prepend: FieldDef::new("answer", T::String),
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            index: 0,
            reason: ApplyError::DuplicateField { ref field, .. },
            ..
        } if field == "answer"
    ));

    let err = parent
        .edited(&[Edit::AugmentSig {
            leaf: parent.root,
            prepend: reasoning_field(),
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::WrongKind { got: "seq", .. },
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// SwapLeaf
// ---------------------------------------------------------------------------

#[test]
fn swap_predict_to_agent_and_back_is_an_involution() {
    let parent = agent_program();
    let summarizer = parent.leaf_id("summarizer").unwrap();
    let calc = tool_id(&parent, "calc");

    let agentic = parent
        .edited(&[Edit::SwapLeaf {
            leaf: summarizer,
            to: SwapTarget::Agent {
                tools: vec![calc],
                stop: StopSpec::default(),
                budget: NodeBudget::default(),
            },
        }])
        .unwrap();

    // Name, sig shape, bindings preserved; kind changed; context slot minted.
    let node = &agentic.nodes[agentic.leaf_id("summarizer").unwrap()];
    let Node::AgentLoop(n) = node else {
        panic!("summarizer should be an agent now, got {node:?}");
    };
    assert_eq!(&*n.tools, &[calc]);
    assert_eq!(agentic.syms.get(n.name), "summarizer");
    assert!(matches!(n.binding[0].src, PortRef::Out { .. }));
    let context = agentic.param_id("summarizer.context").unwrap();
    assert_eq!(agentic.params[context].kind, ParamKind::ContextPolicy);
    // The instruction/demos/model slots carried over, values intact.
    let instr = agentic.param_id("summarizer.instruction").unwrap();
    assert_eq!(
        agentic.params[instr].default,
        ParamValue::Instruction {
            text: "Summarize the answer.".to_string()
        }
    );
    assert_ne!(agentic.meta.program_hash, parent.meta.program_hash);
    assert_round_trips(&agentic);

    // Swap back: the context slot is collected and the content hash returns
    // to the parent's (lineage differs, but lineage is outside the hash).
    let back = agentic
        .edited(&[Edit::SwapLeaf {
            leaf: agentic.leaf_id("summarizer").unwrap(),
            to: SwapTarget::Predict,
        }])
        .unwrap();
    assert!(matches!(
        back.nodes[back.leaf_id("summarizer").unwrap()],
        Node::Predict(_)
    ));
    assert!(back.param_id("summarizer.context").is_none());
    assert_eq!(back.meta.program_hash, parent.meta.program_hash);
    assert_round_trips(&back);
}

#[test]
fn swap_rejects_containers_wrong_directions_and_unknown_tools() {
    let parent = agent_program();
    let researcher = parent.leaf_id("researcher").unwrap();
    let summarizer = parent.leaf_id("summarizer").unwrap();

    let to_agent = SwapTarget::Agent {
        tools: vec![],
        stop: StopSpec::default(),
        budget: NodeBudget::default(),
    };

    // Containers cannot swap.
    let err = parent
        .edited(&[Edit::SwapLeaf {
            leaf: parent.root,
            to: to_agent.clone(),
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::WrongKind { got: "seq", .. },
            ..
        }
    ));

    // Predict → Predict and Agent → Agent are not swaps.
    let err = parent
        .edited(&[Edit::SwapLeaf {
            leaf: summarizer,
            to: SwapTarget::Predict,
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::WrongKind { got: "predict", .. },
            ..
        }
    ));
    let err = parent
        .edited(&[Edit::SwapLeaf {
            leaf: researcher,
            to: to_agent,
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::WrongKind { got: "agent", .. },
            ..
        }
    ));

    // Tools must exist in program.tools.
    let err = parent
        .edited(&[Edit::SwapLeaf {
            leaf: summarizer,
            to: SwapTarget::Agent {
                tools: vec![tid(99)],
                stop: StopSpec::default(),
                budget: NodeBudget::default(),
            },
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::UnknownTool { .. },
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// WrapRetry
// ---------------------------------------------------------------------------

#[test]
fn wrap_retry_rewires_the_parent_and_downstream_ports() {
    let parent = pipeline();
    let drafter = parent.leaf_id("drafter").unwrap();

    let child = parent
        .edited(&[Edit::WrapRetry {
            node: drafter,
            max_attempts: NonZeroU32::new(2).unwrap(),
            backoff_ms: 100,
            feedback: true,
        }])
        .unwrap();

    // The retry sits where the drafter sat; the drafter is its child.
    let new_drafter = child.leaf_id("drafter").unwrap();
    let (retry_id, retry) = child
        .nodes
        .iter()
        .find_map(|(id, node)| match node {
            Node::Retry(r) => Some((id, r)),
            _ => None,
        })
        .expect("a retry node exists");
    assert_eq!(retry.child, new_drafter);
    assert_eq!(retry.max_attempts.get(), 2);
    assert_eq!(retry.backoff_ms, 100);
    assert!(retry.feedback);
    let Node::Seq(root) = &child.nodes[child.root] else {
        panic!("root is a seq");
    };
    assert!(root.body.contains(&retry_id));
    assert!(!root.body.contains(&new_drafter));

    // The checker's binding was redirected to the wrapper (sibling-level
    // visibility) and the whole thing still validates and round-trips.
    let Node::Predict(checker) = &child.nodes[child.leaf_id("checker").unwrap()] else {
        panic!("checker is a predict");
    };
    assert!(matches!(
        checker.binding[0].src,
        PortRef::Out { node, .. } if node == retry_id
    ));
    assert_ne!(child.meta.program_hash, parent.meta.program_hash);
    assert_round_trips(&child);
}

#[test]
fn wrap_retry_on_the_root_fails_validation() {
    let parent = pipeline();
    let err = parent
        .edited(&[Edit::WrapRetry {
            node: parent.root,
            max_attempts: NonZeroU32::new(2).unwrap(),
            backoff_ms: 0,
            feedback: false,
        }])
        .unwrap_err();
    assert!(matches!(err, EditError::Invalid(ValidateError::RootNotSeq)));
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

#[test]
fn remove_collects_the_subtree_and_its_params() {
    // Variant where the checker's output is unused: main exports the draft.
    let mut b = ProgramBuilder::new("removable");
    b.model("m", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let check = b.sig(
        SignatureDef::build("Check")
            .input("answer", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let checker = ir::predict("checker", check).bind("answer", ir::out("drafter", "answer"));
    let parent = b
        .main(
            main_sig,
            ir::seq([drafter, checker]).out("answer", ir::out("drafter", "answer")),
        )
        .unwrap();

    let child = parent
        .edited(&[Edit::Remove {
            node: parent.leaf_id("checker").unwrap(),
        }])
        .unwrap();

    assert!(child.leaf_id("checker").is_none());
    assert_eq!(child.nodes.len(), parent.nodes.len() - 1);
    // The checker's param slots and now-orphaned signature went with it.
    assert!(child.param_id("checker.instruction").is_none());
    assert!(child.param_id("drafter.instruction").is_some());
    assert!(!child.to_dsrs().contains("sig Check"));
    assert_ne!(child.meta.program_hash, parent.meta.program_hash);
    assert_round_trips(&child);
}

#[test]
fn remove_with_a_downstream_reference_fails_validation() {
    let parent = pipeline();
    // checker binds drafter.answer; main exports checker.verdict.
    let err = parent
        .edited(&[Edit::Remove {
            node: parent.leaf_id("drafter").unwrap(),
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Invalid(ValidateError::NodeNotVisible { ref at, .. }) if at == "checker"
    ));
}

#[test]
fn remove_of_the_root_is_rejected() {
    let parent = pipeline();
    let err = parent
        .edited(&[Edit::Remove { node: parent.root }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::NotInSeq { .. },
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// AddTool / RemoveTool / SetStop
// ---------------------------------------------------------------------------

#[test]
fn add_tool_declares_and_remove_tool_clears_stop_tools() {
    let parent = agent_program();
    let researcher = parent.leaf_id("researcher").unwrap();
    let search = tool_id(&parent, "search");
    let calc = tool_id(&parent, "calc");

    let child = parent
        .edited(&[Edit::AddTool {
            agent: researcher,
            tool: calc,
        }])
        .unwrap();
    let Node::AgentLoop(n) = &child.nodes[child.leaf_id("researcher").unwrap()] else {
        panic!("researcher is an agent");
    };
    assert_eq!(&*n.tools, &[search, calc]);
    assert_round_trips(&child);

    // RemoveTool drops the declaration *and* the stop_tools entry.
    let cleared = child
        .edited(&[Edit::RemoveTool {
            agent: child.leaf_id("researcher").unwrap(),
            tool: search,
        }])
        .unwrap();
    let Node::AgentLoop(n) = &cleared.nodes[cleared.leaf_id("researcher").unwrap()] else {
        panic!("researcher is an agent");
    };
    assert_eq!(&*n.tools, &[calc]);
    assert!(n.stop.stop_tools.is_empty());
    assert_round_trips(&cleared);
}

#[test]
fn add_tool_failure_modes() {
    let parent = agent_program();
    let researcher = parent.leaf_id("researcher").unwrap();
    let search = tool_id(&parent, "search");
    let calc = tool_id(&parent, "calc");

    // Unknown ToolId.
    let err = parent
        .edited(&[Edit::AddTool {
            agent: researcher,
            tool: tid(42),
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::UnknownTool { .. },
            ..
        }
    ));

    // Already declared.
    let err = parent
        .edited(&[Edit::AddTool {
            agent: researcher,
            tool: search,
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::ToolAlreadyDeclared { ref name, .. },
            ..
        } if name == "search"
    ));

    // Caps outside the program ceiling. (Builders can never produce this
    // state; simulate a hostile/edited artifact by shrinking the ceiling.)
    let mut stripped = parent.clone();
    stripped.caps = CapSet::new();
    let err = stripped
        .edited(&[Edit::AddTool {
            agent: researcher,
            tool: calc,
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::ToolCapsExceedProgram { ref name, ref missing },
            ..
        } if name == "calc" && missing == &["math:eval".to_string()]
    ));

    // Target must be an agent leaf.
    let err = parent
        .edited(&[Edit::AddTool {
            agent: parent.leaf_id("summarizer").unwrap(),
            tool: calc,
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::WrongKind { got: "predict", .. },
            ..
        }
    ));

    // RemoveTool of an undeclared tool.
    let err = parent
        .edited(&[Edit::RemoveTool {
            agent: researcher,
            tool: calc,
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Apply {
            reason: ApplyError::ToolNotDeclared { ref name, .. },
            ..
        } if name == "calc"
    ));
}

#[test]
fn set_stop_replaces_the_spec_and_is_validated() {
    let parent = agent_program();
    let researcher = parent.leaf_id("researcher").unwrap();
    let search = tool_id(&parent, "search");
    let calc = tool_id(&parent, "calc");

    let stop = StopSpec {
        max_turns: NonZeroU32::new(3).unwrap(),
        stop_tools: Box::new([search]),
        until_parse: false,
    };
    let child = parent
        .edited(&[Edit::SetStop {
            agent: researcher,
            stop: stop.clone(),
        }])
        .unwrap();
    let Node::AgentLoop(n) = &child.nodes[child.leaf_id("researcher").unwrap()] else {
        panic!("researcher is an agent");
    };
    assert_eq!(n.stop, stop);
    assert_round_trips(&child);

    // A stop tool the node does not declare is validate.rs's call.
    let err = parent
        .edited(&[Edit::SetStop {
            agent: researcher,
            stop: StopSpec {
                stop_tools: Box::new([calc]),
                ..StopSpec::default()
            },
        }])
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::Invalid(ValidateError::StopToolNotDeclared { ref at }) if at == "researcher"
    ));
}

// ---------------------------------------------------------------------------
// SetInstructionDefault
// ---------------------------------------------------------------------------

#[test]
fn set_instruction_default_is_a_bake_like_change() {
    let parent = pipeline();
    let child = parent
        .edited(&[Edit::SetInstructionDefault {
            leaf: parent.leaf_id("drafter").unwrap(),
            text: "EDITED: answer in one word.".to_string(),
        }])
        .unwrap();

    let instr = child.param_id("drafter.instruction").unwrap();
    assert_eq!(
        child.params[instr].default,
        ParamValue::Instruction {
            text: "EDITED: answer in one word.".to_string()
        }
    );
    // The parent's default is untouched.
    let parent_instr = parent.param_id("drafter.instruction").unwrap();
    assert_eq!(
        parent.params[parent_instr].default,
        ParamValue::Instruction {
            text: "Answer the question.".to_string()
        }
    );
    assert!(child.to_dsrs().contains("EDITED: answer in one word."));
    assert_ne!(child.meta.program_hash, parent.meta.program_hash);
    assert_round_trips(&child);
}

// ---------------------------------------------------------------------------
// Stale ids, purity, lineage, identity
// ---------------------------------------------------------------------------

#[test]
fn edits_on_stale_node_ids_fail() {
    let parent = pipeline();
    let stale = nid(999);
    let edits = [
        Edit::AugmentSig {
            leaf: stale,
            prepend: reasoning_field(),
        },
        Edit::SwapLeaf {
            leaf: stale,
            to: SwapTarget::Predict,
        },
        Edit::WrapRetry {
            node: stale,
            max_attempts: NonZeroU32::new(2).unwrap(),
            backoff_ms: 0,
            feedback: false,
        },
        Edit::Remove { node: stale },
        Edit::SetStop {
            agent: stale,
            stop: StopSpec::default(),
        },
        Edit::SetInstructionDefault {
            leaf: stale,
            text: "x".to_string(),
        },
    ];
    for edit in edits {
        let err = parent.edited(std::slice::from_ref(&edit)).unwrap_err();
        assert!(
            matches!(
                err,
                EditError::Apply {
                    index: 0,
                    reason: ApplyError::StaleNode { .. },
                    ..
                }
            ),
            "expected StaleNode for {edit:?}, got {err:?}"
        );
    }
}

#[test]
fn edited_is_pure_and_stamps_lineage() {
    let parent = pipeline();
    let parent_hash = parent.meta.program_hash;
    let parent_json = serde_json::to_string(&parent).unwrap();

    let child = parent
        .edited(&[Edit::AugmentSig {
            leaf: parent.leaf_id("drafter").unwrap(),
            prepend: reasoning_field(),
        }])
        .unwrap();

    // Purity: the parent is bit-identical.
    assert_eq!(serde_json::to_string(&parent).unwrap(), parent_json);
    assert_eq!(parent.meta.program_hash, parent_hash);
    assert!(parent.meta.lineage.is_none());

    // The child records its parent the way bake() does, and its hash is the
    // recomputed content hash (lineage is outside the preimage).
    let lineage = child.meta.lineage.as_ref().unwrap();
    assert_eq!(
        lineage.parent.as_deref(),
        Some(format!("{parent_hash:016x}").as_str())
    );
    assert_eq!(child.meta.program_hash, child.compute_hash());
}

#[test]
fn edited_with_no_edits_is_a_hash_no_op() {
    let parent = pipeline();
    let child = parent.edited(&[]).unwrap();
    assert_eq!(child.meta.program_hash, parent.meta.program_hash);
    assert!(child.meta.lineage.is_some());
    assert_round_trips(&child);
}

#[test]
fn a_batch_applies_in_order_over_intermediate_states() {
    let parent = agent_program();
    let summarizer = parent.leaf_id("summarizer").unwrap();
    let researcher = parent.leaf_id("researcher").unwrap();
    let search = tool_id(&parent, "search");
    let calc = tool_id(&parent, "calc");

    // The AddTool/SetStop target the leaf *after* it becomes an agent in the
    // same batch — NodeIds are stable within a batch (swaps are in-place).
    let child = parent
        .edited(&[
            Edit::AugmentSig {
                leaf: researcher,
                prepend: FieldDef::new("plan", T::String),
            },
            Edit::SwapLeaf {
                leaf: summarizer,
                to: SwapTarget::Agent {
                    tools: vec![calc],
                    stop: StopSpec::default(),
                    budget: NodeBudget::default(),
                },
            },
            Edit::AddTool {
                agent: summarizer,
                tool: search,
            },
            Edit::SetStop {
                agent: summarizer,
                stop: StopSpec {
                    max_turns: NonZeroU32::new(4).unwrap(),
                    stop_tools: Box::new([calc]),
                    until_parse: true,
                },
            },
        ])
        .unwrap();

    let Node::AgentLoop(n) = &child.nodes[child.leaf_id("summarizer").unwrap()] else {
        panic!("summarizer became an agent");
    };
    assert_eq!(&*n.tools, &[calc, search]);
    assert_eq!(n.stop.max_turns.get(), 4);
    let Node::AgentLoop(r) = &child.nodes[child.leaf_id("researcher").unwrap()] else {
        panic!("researcher is still an agent");
    };
    assert_eq!(&*child.sigs[r.sig].name, "Research_plan");
    assert_round_trips(&child);
}

// ---------------------------------------------------------------------------
// legal_edits
// ---------------------------------------------------------------------------

#[test]
fn legal_edits_menus_track_node_shape() {
    let p = agent_program();
    let researcher = p.leaf_id("researcher").unwrap();
    let summarizer = p.leaf_id("summarizer").unwrap();
    let search = tool_id(&p, "search");
    let calc = tool_id(&p, "calc");

    let predict_menu = p.legal_edits(summarizer);
    assert!(predict_menu.contains(&EditKind::AugmentSig));
    assert!(predict_menu.contains(&EditKind::SetInstructionDefault));
    assert!(predict_menu.contains(&EditKind::SwapToAgent));
    assert!(predict_menu.contains(&EditKind::WrapRetry));
    assert!(predict_menu.contains(&EditKind::Remove));
    assert!(!predict_menu.contains(&EditKind::SwapToPredict));
    assert!(!predict_menu.contains(&EditKind::SetStop));
    assert!(!predict_menu.contains(&EditKind::AddTool { tool: search }));

    let agent_menu = p.legal_edits(researcher);
    assert!(agent_menu.contains(&EditKind::SwapToPredict));
    assert!(agent_menu.contains(&EditKind::SetStop));
    // Declared tools are removable, undeclared ones addable.
    assert!(agent_menu.contains(&EditKind::RemoveTool { tool: search }));
    assert!(agent_menu.contains(&EditKind::AddTool { tool: calc }));
    assert!(!agent_menu.contains(&EditKind::AddTool { tool: search }));
    assert!(!agent_menu.contains(&EditKind::SwapToAgent));

    // The root cannot be wrapped or removed; a stale id gets no menu.
    assert!(p.legal_edits(p.root).is_empty());
    assert!(p.legal_edits(nid(999)).is_empty());

    // Every menu entry serializes (the menu is proposer-facing data).
    serde_json::to_string(&agent_menu).unwrap();
}

// ---------------------------------------------------------------------------
// migrate_overlay
// ---------------------------------------------------------------------------

fn tuned_overlay(p: &Program) -> Overlay {
    let mut overlay = Overlay::new(p);
    let instr = p.slot_of::<ir::Instruction>("drafter.instruction").unwrap();
    overlay.set_instruction(instr, "TUNED: be terse.");
    let demos = p.slot_of::<ir::Demos>("drafter.demos").unwrap();
    overlay.set_demos(
        demos,
        vec![DemoRow {
            input: obj(&[("question", json!("demo q"))]),
            output: obj(&[("answer", json!("demo a"))]),
        }],
    );
    overlay
}

#[test]
fn migrate_overlay_survives_an_unrelated_edit() {
    let parent = pipeline();
    let overlay = tuned_overlay(&parent);

    // Edit a *different* leaf: the drafter's genes must carry over.
    let child = parent
        .edited(&[Edit::SetInstructionDefault {
            leaf: parent.leaf_id("checker").unwrap(),
            text: "Judge strictly.".to_string(),
        }])
        .unwrap();

    let migrated = migrate_overlay(&parent, &overlay, &child);
    assert_eq!(migrated.base, child.meta.program_hash);
    let instr = child.param_id("drafter.instruction").unwrap();
    assert_eq!(
        migrated.resolve(&child, instr),
        &ParamValue::Instruction {
            text: "TUNED: be terse.".to_string()
        }
    );
    let demos = child.param_id("drafter.demos").unwrap();
    assert!(matches!(
        migrated.resolve(&child, demos),
        ParamValue::Demos { rows } if rows.len() == 1
    ));
}

#[test]
fn migrate_overlay_survives_augment_sig() {
    // Documented decision: AugmentSig keeps inputs identical and only widens
    // the outputs, so demo rows still map onto the base fields — instruction
    // AND demos survive the CoT move.
    let parent = pipeline();
    let overlay = tuned_overlay(&parent);
    let child = parent
        .edited(&[Edit::AugmentSig {
            leaf: parent.leaf_id("drafter").unwrap(),
            prepend: reasoning_field(),
        }])
        .unwrap();

    let migrated = migrate_overlay(&parent, &overlay, &child);
    let instr = child.param_id("drafter.instruction").unwrap();
    let demos = child.param_id("drafter.demos").unwrap();
    assert!(migrated.get(instr).is_some(), "instruction survives");
    assert!(migrated.get(demos).is_some(), "demos survive");
    assert!(matches!(
        migrated.resolve(&child, demos),
        ParamValue::Demos { rows } if rows[0].input.contains_key("question")
    ));
}

#[test]
fn migrate_overlay_drops_entries_when_the_shape_changed() {
    let parent = pipeline();
    let overlay = tuned_overlay(&parent);

    // A child where `drafter` exists at the same ParamPaths but its signature
    // has a different shape (different output field name).
    let mut b = ProgramBuilder::new("pipeline");
    b.model("m", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("reply", T::String)
            .finish()
            .unwrap(),
    );
    let check = b.sig(
        SignatureDef::build("Check")
            .instruction("Judge the answer.")
            .input("answer", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let checker = ir::predict("checker", check).bind("answer", ir::out("drafter", "reply"));
    let child = b
        .main(
            main_sig,
            ir::seq([drafter, checker]).out("verdict", ir::out("checker", "verdict")),
        )
        .unwrap();

    let migrated = migrate_overlay(&parent, &overlay, &child);
    assert_eq!(migrated.base, child.meta.program_hash);
    assert!(
        migrated.is_empty(),
        "shape-changed leaf carries nothing over"
    );
}

#[test]
fn migrate_overlay_remints_model_refs_by_name() {
    // Two models so the ModelRef ordinal is meaningful.
    let build = |name: &str, flip: bool| {
        let mut b = ProgramBuilder::new(name);
        // Registration order differs between parent and child: the ordinal
        // for "deep" is m1 in one and m0 in the other.
        let (fast, deep) = if flip {
            let deep = b.model("deep", model_config("anthropic:claude-sonnet-4-5"));
            let fast = b.model("fast", model_config("openai:gpt-4o-mini"));
            (fast, deep)
        } else {
            let fast = b.model("fast", model_config("openai:gpt-4o-mini"));
            let deep = b.model("deep", model_config("anthropic:claude-sonnet-4-5"));
            (fast, deep)
        };
        let _ = deep;
        let main_sig = b.sig(
            SignatureDef::build("Main")
                .input("question", T::String)
                .output("answer", T::String)
                .finish()
                .unwrap(),
        );
        let node = ir::predict("drafter", main_sig)
            .model(fast)
            .bind("question", ir::input("question"));
        b.main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("drafter", "answer")),
        )
        .unwrap()
    };
    let parent = build("models", false);
    let child = build("models", true);

    let mut overlay = Overlay::new(&parent);
    let model_slot = parent.param_id("drafter.model").unwrap();
    let deep_in_parent = parent
        .models
        .iter()
        .find_map(|(id, m)| (&*m.name == "deep").then_some(id))
        .unwrap();
    overlay
        .set(
            &parent,
            model_slot,
            ParamValue::ModelRef {
                model: deep_in_parent,
            },
        )
        .unwrap();

    let migrated = migrate_overlay(&parent, &overlay, &child);
    let child_slot = child.param_id("drafter.model").unwrap();
    let deep_in_child = child
        .models
        .iter()
        .find_map(|(id, m)| (&*m.name == "deep").then_some(id))
        .unwrap();
    assert_ne!(
        deep_in_parent, deep_in_child,
        "the ordinals genuinely differ"
    );
    assert_eq!(
        migrated.get(child_slot),
        Some(&ParamValue::ModelRef {
            model: deep_in_child
        })
    );
}

#[test]
fn migrate_overlay_with_a_mismatched_base_yields_empty() {
    let parent = pipeline();
    let other = agent_program();
    let overlay = tuned_overlay(&parent);
    // Overlay minted against `parent` cannot be interpreted against `other`.
    let migrated = migrate_overlay(&other, &overlay, &parent);
    assert!(migrated.is_empty());
    assert_eq!(migrated.base, parent.meta.program_hash);
}
