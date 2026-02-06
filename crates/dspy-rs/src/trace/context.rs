use crate::Prediction;
use crate::trace::dag::{Graph, NodeType};
use std::sync::{Arc, Mutex};
use tokio::task_local;
use tracing::{debug, trace};

task_local! {
    static CURRENT_TRACE: Arc<Mutex<Graph>>;
}

#[tracing::instrument(name = "dsrs.trace.scope", level = "debug", skip(f))]
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

pub fn is_tracing() -> bool {
    CURRENT_TRACE.try_with(|_| ()).is_ok()
}

pub fn record_node(
    node_type: NodeType,
    inputs: Vec<usize>,
    input_data: Option<crate::Example>,
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

pub fn record_output(node_id: usize, output: Prediction) {
    let _ = CURRENT_TRACE.try_with(|trace| {
        let mut graph = trace.lock().unwrap();
        graph.set_output(node_id, output);
        trace!(node_id, "trace output recorded");
    });
}
