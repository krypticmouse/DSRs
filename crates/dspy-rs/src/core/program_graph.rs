use std::collections::{HashMap, VecDeque};

use facet::Facet;
use indexmap::IndexMap;

use bamltype::baml_types::{BamlMap, LiteralValue, TypeValue};

use crate::core::{DynModule, PredictState, named_parameters, named_parameters_ref};
use crate::{BamlValue, PredictError, SignatureSchema, TypeIR};

const INPUT_NODE: &str = "input";

pub struct ProgramGraph {
    nodes: IndexMap<String, Node>,
    edges: Vec<Edge>,
}

pub struct Node {
    pub schema: SignatureSchema,
    pub module: Box<dyn DynModule>,
}

impl From<Box<dyn DynModule>> for Node {
    fn from(module: Box<dyn DynModule>) -> Self {
        let schema = module.schema().clone();
        Self { schema, module }
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ProgramGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramGraph")
            .field("nodes", &self.nodes)
            .field("edges", &self.edges)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub from_node: String,
    pub from_field: String,
    pub to_node: String,
    pub to_field: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeAnnotation {
    pub from_node: String,
    pub from_field: String,
    pub to_node: String,
    pub to_field: String,
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("duplicate node `{name}`")]
    DuplicateNode { name: String },
    #[error("missing node `{name}`")]
    MissingNode { name: String },
    #[error("missing field `{field}` on node `{node}` ({side})")]
    MissingField {
        node: String,
        field: String,
        side: &'static str,
    },
    #[error("edge type mismatch `{from_node}.{from_field}` -> `{to_node}.{to_field}`")]
    TypeMismatch {
        from_node: String,
        from_field: String,
        to_node: String,
        to_field: String,
    },
    #[error("graph contains cycle")]
    Cycle,
    #[error("graph has no sink nodes")]
    NoSink,
    #[error("graph has multiple sinks: {sinks:?}")]
    AmbiguousSink { sinks: Vec<String> },
    #[error("projection mismatch at `{path}`: {reason}")]
    ProjectionMismatch { path: String, reason: String },
    #[error("node `{node}` execution failed")]
    Execution {
        node: String,
        #[source]
        source: PredictError,
    },
}

pub trait TypeIrAssignabilityExt {
    fn is_assignable_to(&self, to: &TypeIR) -> bool;
}

impl TypeIrAssignabilityExt for TypeIR {
    fn is_assignable_to(&self, to: &TypeIR) -> bool {
        type_ir_is_assignable(self, to)
    }
}

fn type_ir_is_assignable(from: &TypeIR, to: &TypeIR) -> bool {
    if from == to {
        return true;
    }

    if matches!(to, TypeIR::Top(_)) {
        return true;
    }

    match (from, to) {
        (TypeIR::Literal(from_lit, _), TypeIR::Primitive(to_primitive, _)) => {
            literal_is_assignable_to_primitive(from_lit, to_primitive)
        }
        (TypeIR::Union(from_union, _), _) => from_union
            .iter_include_null()
            .into_iter()
            .all(|from_branch| type_ir_is_assignable(from_branch, to)),
        (_, TypeIR::Union(to_union, _)) => to_union
            .iter_include_null()
            .into_iter()
            .any(|to_branch| type_ir_is_assignable(from, to_branch)),
        (TypeIR::List(from_inner, _), TypeIR::List(to_inner, _)) => {
            type_ir_is_assignable(from_inner, to_inner)
        }
        (TypeIR::Map(from_key, from_value, _), TypeIR::Map(to_key, to_value, _)) => {
            type_ir_is_assignable(from_key, to_key) && type_ir_is_assignable(from_value, to_value)
        }
        (TypeIR::Tuple(from_items, _), TypeIR::Tuple(to_items, _)) => {
            from_items.len() == to_items.len()
                && from_items
                    .iter()
                    .zip(to_items)
                    .all(|(from_item, to_item)| type_ir_is_assignable(from_item, to_item))
        }
        (
            TypeIR::Class {
                name: from_name,
                dynamic: from_dynamic,
                ..
            },
            TypeIR::Class {
                name: to_name,
                dynamic: to_dynamic,
                ..
            },
        ) => from_name == to_name && from_dynamic == to_dynamic,
        (
            TypeIR::Enum {
                name: from_name,
                dynamic: from_dynamic,
                ..
            },
            TypeIR::Enum {
                name: to_name,
                dynamic: to_dynamic,
                ..
            },
        ) => from_name == to_name && from_dynamic == to_dynamic,
        (
            TypeIR::RecursiveTypeAlias {
                name: from_name,
                mode: from_mode,
                ..
            },
            TypeIR::RecursiveTypeAlias {
                name: to_name,
                mode: to_mode,
                ..
            },
        ) => from_name == to_name && from_mode == to_mode,
        _ => false,
    }
}

fn literal_is_assignable_to_primitive(from: &LiteralValue, to: &TypeValue) -> bool {
    matches!(
        (from, to),
        (LiteralValue::String(_), TypeValue::String)
            | (LiteralValue::Int(_), TypeValue::Int)
            | (LiteralValue::Bool(_), TypeValue::Bool)
    )
}

fn projection_mismatch(path: impl Into<String>, reason: impl Into<String>) -> GraphError {
    GraphError::ProjectionMismatch {
        path: path.into(),
        reason: reason.into(),
    }
}

fn missing_node(name: &str) -> GraphError {
    GraphError::MissingNode {
        name: name.to_string(),
    }
}

fn missing_field(node: &str, field: &str, side: &'static str) -> GraphError {
    GraphError::MissingField {
        node: node.to_string(),
        field: field.to_string(),
        side,
    }
}

fn sync_node_schema(node: &mut Node) {
    // Keep schema/module in sync even when callers manually construct Node.
    node.schema = node.module.schema().clone();
}

impl ProgramGraph {
    pub fn new() -> Self {
        Self {
            nodes: IndexMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn nodes(&self) -> &IndexMap<String, Node> {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut IndexMap<String, Node> {
        &mut self.nodes
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        node: impl Into<Node>,
    ) -> Result<(), GraphError> {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            return Err(GraphError::DuplicateNode { name });
        }
        let mut node = node.into();
        sync_node_schema(&mut node);
        self.nodes.insert(name, node);
        Ok(())
    }

    pub fn remove_node(&mut self, name: &str) -> Result<Node, GraphError> {
        let removed = self
            .nodes
            .shift_remove(name)
            .ok_or_else(|| missing_node(name))?;
        self.edges
            .retain(|edge| edge.from_node != name && edge.to_node != name);
        Ok(removed)
    }

    pub fn connect(
        &mut self,
        from: &str,
        from_field: &str,
        to: &str,
        to_field: &str,
    ) -> Result<(), GraphError> {
        self.validate_edge(from, from_field, to, to_field)?;
        self.edges.push(Edge {
            from_node: from.to_string(),
            from_field: from_field.to_string(),
            to_node: to.to_string(),
            to_field: to_field.to_string(),
        });
        Ok(())
    }

    pub fn replace_node(&mut self, name: &str, node: impl Into<Node>) -> Result<(), GraphError> {
        if !self.nodes.contains_key(name) {
            return Err(missing_node(name));
        }
        let mut node = node.into();
        sync_node_schema(&mut node);

        let incident = self
            .edges
            .iter()
            .filter(|edge| edge.from_node == name || edge.to_node == name)
            .cloned()
            .collect::<Vec<_>>();

        let old = self
            .nodes
            .insert(name.to_string(), node)
            .expect("node existence checked");

        for edge in incident {
            if let Err(err) = self.validate_edge(
                &edge.from_node,
                &edge.from_field,
                &edge.to_node,
                &edge.to_field,
            ) {
                self.nodes.insert(name.to_string(), old);
                return Err(err);
            }
        }

        Ok(())
    }

    pub fn insert_between(
        &mut self,
        from: &str,
        to: &str,
        inserted_name: impl Into<String>,
        inserted_node: Node,
        from_field: &str,
        to_field: &str,
    ) -> Result<(), GraphError> {
        let inserted_name = inserted_name.into();
        if self.nodes.contains_key(&inserted_name) {
            return Err(GraphError::DuplicateNode {
                name: inserted_name,
            });
        }

        let edge_index = self
            .edges
            .iter()
            .position(|edge| {
                edge.from_node == from
                    && edge.to_node == to
                    && edge.from_field == from_field
                    && edge.to_field == to_field
            })
            .ok_or_else(|| {
                projection_mismatch(
                    format!("{from}.{from_field}->{to}.{to_field}"),
                    "edge not found for insert_between",
                )
            })?;

        let inserted_input = inserted_node
            .schema
            .input_fields()
            .first()
            .ok_or_else(|| projection_mismatch(inserted_name.clone(), "inserted node has no input fields"))?
            .rust_name
            .clone();
        let inserted_output = inserted_node
            .schema
            .output_fields()
            .first()
            .ok_or_else(|| {
                projection_mismatch(inserted_name.clone(), "inserted node has no output fields")
            })?
            .rust_name
            .clone();

        self.nodes.insert(inserted_name.clone(), inserted_node);

        let direct_edge = self.edges.remove(edge_index);

        if let Err(err) = self.connect(
            &direct_edge.from_node,
            &direct_edge.from_field,
            &inserted_name,
            &inserted_input,
        ) {
            self.nodes.shift_remove(&inserted_name);
            self.edges.insert(edge_index, direct_edge);
            return Err(err);
        }

        if let Err(err) = self.connect(
            &inserted_name,
            &inserted_output,
            &direct_edge.to_node,
            &direct_edge.to_field,
        ) {
            self.nodes.shift_remove(&inserted_name);
            self.edges.retain(|edge| {
                !(edge.from_node == direct_edge.from_node
                    && edge.to_node == inserted_name
                    && edge.from_field == direct_edge.from_field
                    && edge.to_field == inserted_input)
            });
            self.edges.insert(edge_index, direct_edge);
            return Err(err);
        }

        Ok(())
    }

    pub async fn execute(&self, input: BamlValue) -> Result<BamlValue, GraphError> {
        let order = self.topological_order()?;
        let mut outputs: HashMap<String, BamlValue> = HashMap::new();

        for node_name in &order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| missing_node(node_name))?;

            let incoming = self
                .edges
                .iter()
                .filter(|edge| edge.to_node == *node_name)
                .collect::<Vec<_>>();

            let node_input =
                if incoming.is_empty() {
                    input.clone()
                } else {
                    let mut map = BamlMap::new();
                    for edge in incoming {
                        if edge.from_node == INPUT_NODE {
                            let value = navigate_runtime_path(&input, &edge.from_field)
                                .ok_or_else(|| {
                                    projection_mismatch(
                                        format!("{INPUT_NODE}.{}", edge.from_field),
                                        "source value missing",
                                    )
                                })?;
                            let to_schema = find_input_field(&node.schema, &edge.to_field)
                                .ok_or_else(|| missing_field(&edge.to_node, &edge.to_field, "input"))?;
                            insert_baml_at_path(&mut map, to_schema.path(), value.clone());
                            continue;
                        }

                        let upstream = outputs
                            .get(&edge.from_node)
                            .ok_or_else(|| projection_mismatch(&edge.from_node, "missing upstream output"))?;
                        let from_node = self
                            .nodes
                            .get(&edge.from_node)
                            .ok_or_else(|| missing_node(&edge.from_node))?;
                        let from_schema = find_output_field(&from_node.schema, &edge.from_field)
                            .ok_or_else(|| missing_field(&edge.from_node, &edge.from_field, "output"))?;
                        let value = from_node
                            .schema
                            .navigate_field(from_schema.path(), upstream)
                            .ok_or_else(|| {
                                projection_mismatch(
                                    format!("{}.{}", edge.from_node, edge.from_field),
                                    "source value missing",
                                )
                            })?
                            .clone();

                        let to_schema = find_input_field(&node.schema, &edge.to_field)
                            .ok_or_else(|| missing_field(&edge.to_node, &edge.to_field, "input"))?;

                        insert_baml_at_path(&mut map, to_schema.path(), value);
                    }
                    BamlValue::Class("GraphInput".to_string(), map)
                };

            let predicted =
                node.module
                    .forward(node_input)
                    .await
                    .map_err(|source| GraphError::Execution {
                        node: node_name.clone(),
                        source,
                    })?;
            outputs.insert(node_name.clone(), predicted.into_inner());
        }

        let sinks = self.sink_nodes();
        match sinks.len() {
            0 => Err(GraphError::NoSink),
            1 => outputs
                .remove(&sinks[0])
                .ok_or_else(|| projection_mismatch(sinks[0].clone(), "sink output missing")),
            _ => Err(GraphError::AmbiguousSink { sinks }),
        }
    }

    pub fn from_module<M>(module: &M) -> Result<Self, GraphError>
    where
        M: for<'a> Facet<'a>,
    {
        Self::from_module_with_annotations(module, &[])
    }

    pub fn from_module_with_annotations<M>(
        module: &M,
        annotations: &[GraphEdgeAnnotation],
    ) -> Result<Self, GraphError>
    where
        M: for<'a> Facet<'a>,
    {
        let mut graph = ProgramGraph::new();
        let predictors =
            named_parameters_ref(module).map_err(|err| projection_mismatch("<module>", err.to_string()))?;

        for (path, predictor) in predictors {
            let state = predictor.dump_state();

            let mut dyn_module: Box<dyn DynModule> =
                Box::new(crate::core::PredictDynModule::new(predictor.schema().clone()));
            let leaves = dyn_module.predictors_mut();
            let Some((_, dyn_predictor)) = leaves.into_iter().next() else {
                return Err(projection_mismatch(path, "dynamic module has no predictor leaves"));
            };
            dyn_predictor
                .load_state(state)
                .map_err(|err| projection_mismatch(path.clone(), err.to_string()))?;

            graph.add_node(path, dyn_module)?;
        }

        for annotation in annotations {
            graph.connect(
                &annotation.from_node,
                &annotation.from_field,
                &annotation.to_node,
                &annotation.to_field,
            )?;
        }

        if graph.edges.is_empty() {
            graph.infer_edges_by_schema_order()?;
        }
        if graph.nodes.len() > 1 && graph.edges.is_empty() {
            return Err(projection_mismatch(
                "<module>",
                "projection produced multiple nodes with no resolvable edges",
            ));
        }

        Ok(graph)
    }

    pub fn fit<M>(&self, module: &mut M) -> Result<(), GraphError>
    where
        M: for<'a> Facet<'a>,
    {
        let mut destination =
            named_parameters(module).map_err(|err| projection_mismatch("<module>", err.to_string()))?;

        for (node_name, node) in &self.nodes {
            let mut node_predictors = node.module.predictors();
            let Some((_, predictor)) = node_predictors.pop() else {
                continue;
            };
            let state: PredictState = predictor.dump_state();

            let Some((_, target)) = destination.iter_mut().find(|(path, _)| path == node_name)
            else {
                return Err(projection_mismatch(
                    node_name.clone(),
                    "graph node has no matching typed predictor path",
                ));
            };
            target
                .load_state(state)
                .map_err(|err| projection_mismatch(node_name.clone(), err.to_string()))?;
        }

        Ok(())
    }

    fn infer_edges_by_schema_order(&mut self) -> Result<(), GraphError> {
        let node_names = self.nodes.keys().cloned().collect::<Vec<_>>();
        let mut inferred = Vec::<(String, String, String, String)>::new();

        for from_idx in 0..node_names.len() {
            for to_idx in (from_idx + 1)..node_names.len() {
                let from_name = &node_names[from_idx];
                let to_name = &node_names[to_idx];
                let from_schema = &self
                    .nodes
                    .get(from_name)
                    .expect("node names collected from map")
                    .schema;
                let to_schema = &self
                    .nodes
                    .get(to_name)
                    .expect("node names collected from map")
                    .schema;

                for from_field in from_schema.output_fields() {
                    for to_field in to_schema.input_fields() {
                        let names_match = from_field.rust_name == to_field.rust_name
                            || from_field.lm_name == to_field.lm_name;
                        if !names_match {
                            continue;
                        }
                        if !from_field.type_ir.is_assignable_to(&to_field.type_ir) {
                            continue;
                        }
                        if self.edges.iter().any(|edge| {
                            edge.from_node == *from_name
                                && edge.from_field == from_field.rust_name
                                && edge.to_node == *to_name
                                && edge.to_field == to_field.rust_name
                        }) {
                            continue;
                        }
                        inferred.push((
                            from_name.clone(),
                            from_field.rust_name.clone(),
                            to_name.clone(),
                            to_field.rust_name.clone(),
                        ));
                    }
                }
            }
        }

        for (from_node, from_field, to_node, to_field) in inferred {
            self.connect(&from_node, &from_field, &to_node, &to_field)?;
        }
        Ok(())
    }

    fn validate_edge(
        &self,
        from: &str,
        from_field: &str,
        to: &str,
        to_field: &str,
    ) -> Result<(), GraphError> {
        let to_node = self.nodes.get(to).ok_or_else(|| missing_node(to))?;
        let to_schema = find_input_field(&to_node.schema, to_field)
            .ok_or_else(|| missing_field(to, to_field, "input"))?;

        if from == INPUT_NODE {
            if from_field.trim().is_empty() {
                return Err(projection_mismatch(
                    format!("{INPUT_NODE}.{from_field}"),
                    "input edge field cannot be empty",
                ));
            }
            return Ok(());
        }

        let from_node = self
            .nodes
            .get(from)
            .ok_or_else(|| missing_node(from))?;
        let from_schema = find_output_field(&from_node.schema, from_field)
            .ok_or_else(|| missing_field(from, from_field, "output"))?;

        if !from_schema.type_ir.is_assignable_to(&to_schema.type_ir) {
            return Err(GraphError::TypeMismatch {
                from_node: from.to_string(),
                from_field: from_field.to_string(),
                to_node: to.to_string(),
                to_field: to_field.to_string(),
            });
        }

        Ok(())
    }

    fn topological_order(&self) -> Result<Vec<String>, GraphError> {
        let mut indegree: HashMap<&str, usize> = self
            .nodes
            .keys()
            .map(|name| (name.as_str(), 0usize))
            .collect();

        for edge in &self.edges {
            if edge.from_node == INPUT_NODE {
                if !self.nodes.contains_key(&edge.to_node) {
                    return Err(missing_node(&edge.to_node));
                }
                continue;
            }
            if !self.nodes.contains_key(&edge.from_node) {
                return Err(missing_node(&edge.from_node));
            }
            if !self.nodes.contains_key(&edge.to_node) {
                return Err(missing_node(&edge.to_node));
            }
            *indegree
                .get_mut(edge.to_node.as_str())
                .expect("to_node existence checked") += 1;
        }

        let mut queue = VecDeque::new();
        for name in self.nodes.keys() {
            if indegree[name.as_str()] == 0 {
                queue.push_back(name.clone());
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(node) = queue.pop_front() {
            order.push(node.clone());
            for edge in self.edges.iter().filter(|edge| edge.from_node == node) {
                let target = edge.to_node.as_str();
                let current = indegree.get_mut(target).expect("target should exist");
                *current -= 1;
                if *current == 0 {
                    queue.push_back(edge.to_node.clone());
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::Cycle);
        }

        Ok(order)
    }

    fn sink_nodes(&self) -> Vec<String> {
        let mut outgoing = HashMap::<&str, usize>::new();
        for name in self.nodes.keys() {
            outgoing.insert(name, 0);
        }
        for edge in &self.edges {
            if let Some(count) = outgoing.get_mut(edge.from_node.as_str()) {
                *count += 1;
            }
        }

        self.nodes
            .keys()
            .filter(|name| outgoing.get(name.as_str()).copied().unwrap_or(0) == 0)
            .cloned()
            .collect()
    }
}

impl Default for ProgramGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn find_input_field<'a>(
    schema: &'a SignatureSchema,
    field: &str,
) -> Option<&'a crate::FieldSchema> {
    schema
        .input_fields()
        .iter()
        .find(|candidate| candidate.rust_name == field || candidate.lm_name == field)
}

fn find_output_field<'a>(
    schema: &'a SignatureSchema,
    field: &str,
) -> Option<&'a crate::FieldSchema> {
    schema
        .output_fields()
        .iter()
        .find(|candidate| candidate.rust_name == field || candidate.lm_name == field)
}

fn navigate_runtime_path<'a>(root: &'a BamlValue, field_path: &str) -> Option<&'a BamlValue> {
    let mut current = root;
    for part in field_path.split('.').filter(|part| !part.is_empty()) {
        current = match current {
            BamlValue::Class(_, map) | BamlValue::Map(map) => map.get(part)?,
            _ => return None,
        };
    }
    Some(current)
}

fn insert_baml_at_path(
    root: &mut BamlMap<String, BamlValue>,
    path: &crate::FieldPath,
    value: BamlValue,
) {
    let parts: Vec<_> = path.iter().collect();
    if parts.is_empty() {
        return;
    }
    insert_baml_at_parts(root, &parts, value);
}

fn insert_baml_at_parts(
    root: &mut BamlMap<String, BamlValue>,
    parts: &[&'static str],
    value: BamlValue,
) {
    if parts.len() == 1 {
        root.insert(parts[0].to_string(), value);
        return;
    }

    let key = parts[0].to_string();
    let entry = root
        .entry(key)
        .or_insert_with(|| BamlValue::Map(BamlMap::new()));

    if !matches!(entry, BamlValue::Map(_) | BamlValue::Class(_, _)) {
        *entry = BamlValue::Map(BamlMap::new());
    }

    let child = match entry {
        BamlValue::Map(map) | BamlValue::Class(_, map) => map,
        _ => unreachable!(),
    };

    insert_baml_at_parts(child, &parts[1..], value);
}
