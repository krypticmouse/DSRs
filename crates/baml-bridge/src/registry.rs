use std::collections::{HashMap, HashSet};

use baml_types::{type_meta, StreamingMode, TypeIR};
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum, OutputFormatContent};

use crate::prompt::renderer::{RendererDbSeed, RendererKey, RendererSpec};

#[derive(Debug, Default)]
pub struct Registry {
    enums: IndexMap<String, Enum>,
    classes: IndexMap<(String, StreamingMode), Class>,
    class_deps: IndexMap<String, IndexSet<String>>,
    structural_recursive_aliases: IndexMap<String, TypeIR>,
    registered: HashSet<String>,
    renderers: RendererDbSeed,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_type(&mut self, name: &str) -> bool {
        if self.registered.contains(name) {
            return false;
        }
        self.registered.insert(name.to_string());
        true
    }

    pub fn register_enum(&mut self, r#enum: Enum) {
        let name = r#enum.name.real_name().to_string();
        self.enums.entry(name).or_insert(r#enum);
    }

    pub fn register_class(&mut self, class: Class) {
        let name = class.name.real_name().to_string();
        let mode = class.namespace;
        let key = (name.clone(), mode);
        let entry = self.classes.entry(key).or_insert_with(|| class);

        let deps = self.class_deps.entry(name).or_default();
        for (_, field_type, _, _) in &entry.fields {
            collect_class_refs(field_type, deps);
        }
    }

    pub fn register_structural_alias(&mut self, name: String, alias: TypeIR) {
        self.structural_recursive_aliases.insert(name, alias);
    }

    pub fn register_renderer(&mut self, key: RendererKey, spec: RendererSpec) {
        self.renderers.insert(key, spec);
    }

    pub fn build_with_renderers(self, target: TypeIR) -> (OutputFormatContent, RendererDbSeed) {
        let recursive_classes = compute_recursive_classes(&self.class_deps);

        let mut enums = self.enums.into_iter().collect::<Vec<_>>();
        enums.sort_by(|(a, _), (b, _)| a.cmp(b));
        let enums = enums.into_iter().map(|(_, v)| v).collect::<Vec<_>>();

        let mut classes = self.classes.into_iter().collect::<Vec<_>>();
        classes.sort_by(|(a, _), (b, _)| {
            let (a_name, a_mode) = a;
            let (b_name, b_mode) = b;
            match a_name.cmp(b_name) {
                std::cmp::Ordering::Equal => mode_rank(*a_mode).cmp(&mode_rank(*b_mode)),
                other => other,
            }
        });
        let classes = classes.into_iter().map(|(_, v)| v).collect::<Vec<_>>();

        let output_format = OutputFormatContent::target(target)
            .enums(enums)
            .classes(classes)
            .recursive_classes(recursive_classes)
            .structural_recursive_aliases(self.structural_recursive_aliases)
            .build();

        (output_format, self.renderers)
    }

    pub fn build(self, target: TypeIR) -> OutputFormatContent {
        self.build_with_renderers(target).0
    }
}

fn collect_class_refs(field_type: &TypeIR, deps: &mut IndexSet<String>) {
    match field_type {
        TypeIR::Class { name, .. } => {
            deps.insert(name.clone());
        }
        TypeIR::RecursiveTypeAlias { name, .. } => {
            deps.insert(name.clone());
        }
        TypeIR::List(inner, _) => collect_class_refs(inner, deps),
        TypeIR::Map(key, value, _) => {
            collect_class_refs(key, deps);
            collect_class_refs(value, deps);
        }
        TypeIR::Union(union, _) => {
            for item in union.iter_include_null() {
                collect_class_refs(item, deps);
            }
        }
        TypeIR::Tuple(items, _) => {
            for item in items {
                collect_class_refs(item, deps);
            }
        }
        TypeIR::Arrow(arrow, _) => {
            for param in &arrow.param_types {
                collect_class_refs(param, deps);
            }
            collect_class_refs(&arrow.return_type, deps);
        }
        TypeIR::Primitive(..) | TypeIR::Enum { .. } | TypeIR::Literal(..) | TypeIR::Top(..) => {}
    }
}

fn compute_recursive_classes(class_deps: &IndexMap<String, IndexSet<String>>) -> IndexSet<String> {
    let mut index = 0usize;
    let mut indices: HashMap<String, usize> = HashMap::new();
    let mut lowlink: HashMap<String, usize> = HashMap::new();
    let mut stack: Vec<String> = Vec::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut recursive = HashSet::new();

    #[allow(clippy::too_many_arguments)]
    fn strongconnect(
        node: &str,
        index: &mut usize,
        indices: &mut HashMap<String, usize>,
        lowlink: &mut HashMap<String, usize>,
        stack: &mut Vec<String>,
        on_stack: &mut HashSet<String>,
        deps: &IndexMap<String, IndexSet<String>>,
        recursive: &mut HashSet<String>,
    ) {
        let node_key = node.to_string();
        indices.insert(node_key.clone(), *index);
        lowlink.insert(node_key.clone(), *index);
        *index += 1;
        stack.push(node_key.clone());
        on_stack.insert(node_key.clone());

        if let Some(children) = deps.get(node) {
            for child in children.iter() {
                if !indices.contains_key(child) {
                    strongconnect(
                        child, index, indices, lowlink, stack, on_stack, deps, recursive,
                    );
                    let lowlink_child = lowlink.get(child).copied().unwrap_or(0);
                    let lowlink_node = lowlink.get(&node_key).copied().unwrap_or(0);
                    lowlink.insert(node_key.clone(), lowlink_node.min(lowlink_child));
                } else if on_stack.contains(child) {
                    let index_child = indices.get(child).copied().unwrap_or(0);
                    let lowlink_node = lowlink.get(&node_key).copied().unwrap_or(0);
                    lowlink.insert(node_key.clone(), lowlink_node.min(index_child));
                }
            }
        }

        let node_index = indices.get(&node_key).copied().unwrap_or(0);
        let node_lowlink = lowlink.get(&node_key).copied().unwrap_or(0);
        if node_lowlink == node_index {
            let mut scc = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == node_key {
                    break;
                }
            }

            if scc.len() > 1 {
                for name in scc {
                    recursive.insert(name);
                }
            } else if let Some(name) = scc.first() {
                if deps
                    .get(name)
                    .map(|edges| edges.contains(name))
                    .unwrap_or(false)
                {
                    recursive.insert(name.clone());
                }
            }
        }
    }

    for node in class_deps.keys() {
        if !indices.contains_key(node) {
            strongconnect(
                node,
                &mut index,
                &mut indices,
                &mut lowlink,
                &mut stack,
                &mut on_stack,
                class_deps,
                &mut recursive,
            );
        }
    }

    let mut sorted = recursive.into_iter().collect::<Vec<_>>();
    sorted.sort();
    IndexSet::from_iter(sorted)
}

pub fn default_streaming_behavior() -> type_meta::base::StreamingBehavior {
    type_meta::base::StreamingBehavior::default()
}

fn mode_rank(mode: StreamingMode) -> u8 {
    match mode {
        StreamingMode::NonStreaming => 0,
        StreamingMode::Streaming => 1,
    }
}
