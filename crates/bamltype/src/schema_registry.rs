//! Schema registry for collected BAML classes/enums during shape traversal.

use std::collections::{HashMap, HashSet};

use baml_types::{StreamingMode, TypeIR};
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum, OutputFormatContent};

/// Registry for schema elements generated from facet shapes.
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    enums: IndexMap<String, Enum>,
    classes: IndexMap<(String, StreamingMode), Class>,
    class_deps: IndexMap<String, IndexSet<String>>,
    structural_recursive_aliases: IndexMap<String, TypeIR>,
    registered: HashSet<String>,
}

impl SchemaRegistry {
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
        let entry = self.classes.entry(key).or_insert(class);

        let deps = self.class_deps.entry(name).or_default();
        for (_, field_type, _, _) in &entry.fields {
            collect_class_refs(field_type, deps);
        }
    }

    pub fn register_structural_alias(&mut self, name: String, alias: TypeIR) {
        self.structural_recursive_aliases.insert(name, alias);
    }

    pub fn build(self, target: TypeIR) -> OutputFormatContent {
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

        OutputFormatContent::target(target)
            .enums(enums)
            .classes(classes)
            .recursive_classes(recursive_classes)
            .structural_recursive_aliases(self.structural_recursive_aliases)
            .build()
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
    struct Tarjan<'a> {
        next_index: usize,
        indices: HashMap<String, usize>,
        lowlink: HashMap<String, usize>,
        stack: Vec<String>,
        on_stack: HashSet<String>,
        deps: &'a IndexMap<String, IndexSet<String>>,
        recursive: HashSet<String>,
    }

    impl<'a> Tarjan<'a> {
        fn new(deps: &'a IndexMap<String, IndexSet<String>>) -> Self {
            Self {
                next_index: 0,
                indices: HashMap::new(),
                lowlink: HashMap::new(),
                stack: Vec::new(),
                on_stack: HashSet::new(),
                deps,
                recursive: HashSet::new(),
            }
        }

        fn strongconnect(&mut self, node: &str) {
            let node_key = node.to_string();
            self.indices.insert(node_key.clone(), self.next_index);
            self.lowlink.insert(node_key.clone(), self.next_index);
            self.next_index += 1;
            self.stack.push(node_key.clone());
            self.on_stack.insert(node_key.clone());

            let child_names = self
                .deps
                .get(node)
                .map(|children| children.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            for child in child_names {
                if !self.indices.contains_key(&child) {
                    self.strongconnect(&child);
                    let lowlink_child = self.lowlink.get(&child).copied().unwrap_or(0);
                    let lowlink_node = self.lowlink.get(&node_key).copied().unwrap_or(0);
                    self.lowlink
                        .insert(node_key.clone(), lowlink_node.min(lowlink_child));
                } else if self.on_stack.contains(&child) {
                    let index_child = self.indices.get(&child).copied().unwrap_or(0);
                    let lowlink_node = self.lowlink.get(&node_key).copied().unwrap_or(0);
                    self.lowlink
                        .insert(node_key.clone(), lowlink_node.min(index_child));
                }
            }

            let node_index = self.indices.get(&node_key).copied().unwrap_or(0);
            let node_lowlink = self.lowlink.get(&node_key).copied().unwrap_or(0);
            if node_lowlink == node_index {
                let mut scc = Vec::new();
                while let Some(w) = self.stack.pop() {
                    self.on_stack.remove(&w);
                    scc.push(w.clone());
                    if w == node_key {
                        break;
                    }
                }

                if scc.len() > 1 {
                    for name in scc {
                        self.recursive.insert(name);
                    }
                } else if let Some(name) = scc.first()
                    && self
                        .deps
                        .get(name)
                        .map(|edges| edges.contains(name))
                        .unwrap_or(false)
                {
                    self.recursive.insert(name.clone());
                }
            }
        }
    }

    let mut tarjan = Tarjan::new(class_deps);
    for node in class_deps.keys() {
        if !tarjan.indices.contains_key(node) {
            tarjan.strongconnect(node);
        }
    }

    let mut sorted = tarjan.recursive.into_iter().collect::<Vec<_>>();
    sorted.sort();
    IndexSet::from_iter(sorted)
}

fn mode_rank(mode: StreamingMode) -> u8 {
    match mode {
        StreamingMode::NonStreaming => 0,
        StreamingMode::Streaming => 1,
    }
}
