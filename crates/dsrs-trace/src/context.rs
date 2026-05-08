use crate::dag::{Graph, NodeType};
use dsrs_core::{Prediction, RawExample};
use std::sync::{Arc, Mutex};
use tokio::task_local;
use tracing::{debug, trace};

task_local! {
    static CURRENT_TRACE: Arc<Mutex<Graph>>;
}

#[tracing::instrument(name = "dsrs.trace.scope", level = "debug", skip(f))]
/// Runs an async closure while recording all [`Predict`](crate::Predict) calls into a
/// computation [`Graph`].
///
/// Returns the closure's result and the recorded graph. Uses `tokio::task_local!` for
/// scoping — only calls on the same task see the trace context. Spawned subtasks
/// will NOT be traced unless they inherit the task-local.
pub async fn trace<F, Fut, R>(f: F) -> (R, Graph)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let graph = Arc::new(Mutex::new(Graph::new()));
    debug!("trace scope started");
    let result = CURRENT_TRACE.scope(graph.clone(), f()).await;

    // We need to unwrap the graph.
    // If there are other references (which shouldn't be if scope ended and we are the only owner of the Arc),
    // try_unwrap works.
    // However, if tasks are still running (orphaned), this might fail.
    // Assuming well-behaved usage.
    let graph = match Arc::try_unwrap(graph) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(), // Fallback: clone if still shared
    };
    debug!(node_count = graph.nodes.len(), "trace scope completed");

    (result, graph)
}

/// Returns `true` if the current task is inside a [`trace()`] scope.
///
/// Used internally by [`Predict`](crate::Predict) to decide whether to record nodes.
/// You can also use it to conditionally enable expensive debug logging.
pub fn is_tracing() -> bool {
    CURRENT_TRACE.try_with(|_| ()).is_ok()
}

/// Records a node in the current trace graph. Returns the node ID, or `None` if
/// not inside a [`trace()`] scope.
///
/// Called internally by [`Predict::forward`](crate::Predict) — you don't call this directly
/// unless you're implementing a custom module that needs trace integration.
pub fn record_node(
    node_type: NodeType,
    inputs: Vec<usize>,
    input_data: Option<RawExample>,
) -> Option<usize> {
    let input_count = inputs.len();
    let has_input_data = input_data.is_some();
    CURRENT_TRACE
        .try_with(|trace| {
            let mut graph = trace.lock().unwrap();
            let node_id = graph.add_node(node_type.clone(), inputs, input_data);
            trace!(
                node_id,
                ?node_type,
                input_count,
                has_input_data,
                "trace node recorded"
            );
            Some(node_id)
        })
        .unwrap_or(None)
}

/// Attaches output data to a previously recorded trace node.
///
/// Called internally after a [`Predict`](crate::Predict) call completes. No-op if
/// not inside a [`trace()`] scope.
pub fn record_output(node_id: usize, output: Prediction) {
    let _ = CURRENT_TRACE.try_with(|trace| {
        let mut graph = trace.lock().unwrap();
        graph.set_output(node_id, output);
        trace!(node_id, "trace output recorded");
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsrs_core::{LmUsage, hashmap};

    #[tokio::test]
    async fn trace_scope_records_nodes_and_outputs() {
        assert!(!is_tracing());

        let (result, graph) = trace(|| async {
            assert!(is_tracing());
            let id = record_node(
                NodeType::Operator {
                    name: "normalize".to_string(),
                },
                vec![],
                None,
            )
            .unwrap();
            record_output(
                id,
                Prediction::new(
                    hashmap! {
                        "normalized".to_string() => true.into(),
                    },
                    LmUsage::default(),
                ),
            );
            7
        })
        .await;

        assert_eq!(result, 7);
        assert!(!is_tracing());
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].output.as_ref().unwrap()["normalized"], true);
    }

    #[test]
    fn record_outside_trace_is_noop() {
        assert!(record_node(NodeType::Root, vec![], None).is_none());
        record_output(0, Prediction::new(hashmap! {}, LmUsage::default()));
    }
}
