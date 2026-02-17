use std::collections::{BTreeSet, HashMap, VecDeque};

use bytes::BufMut;
use futures::{future::BoxFuture, stream::FuturesUnordered, FutureExt, StreamExt};
use hive_router_internal::telemetry::traces::spans::graphql::{
    GraphQLOperationSpan, GraphQLSpanOperationIdentity, GraphQLSubgraphOperationSpan,
};
use hive_router_query_planner::{
    planner::plan_nodes::{
        ConditionNode, FetchNode, FetchRewrite, FlattenNodePath, PlanNode, QueryPlan,
    },
    state::supergraph_state::OperationKind,
};
use http::HeaderMap;
use sonic_rs::ValueRef;
use tracing::Instrument;

use crate::{
    context::ExecutionContext,
    execution::{
        client_request_details::ClientRequestDetails,
        error::{IntoPlanExecutionError, LazyPlanContext, PlanExecutionError},
        jwt_forward::JwtAuthForwardingPlan,
        rewrites::FetchRewriteExt,
    },
    executors::{common::SubgraphExecutionRequest, map::SubgraphExecutorMap},
    headers::{
        plan::{HeaderRulesPlan, ResponseHeaderAggregator},
        request::modify_subgraph_request_headers,
        response::apply_subgraph_response_headers,
    },
    introspection::{
        resolve::{resolve_introspection, IntrospectionContext},
        schema::SchemaMetadata,
    },
    projection::{
        plan::FieldProjectionPlan, request::project_requires, response::project_by_operation,
    },
    response::{
        graphql_error::{GraphQLError, GraphQLErrorPath},
        merge::deep_merge,
        subgraph_response::SubgraphResponse,
        value::Value,
    },
    utils::{
        consts::{CLOSE_BRACKET, OPEN_BRACKET},
        traverse::{traverse_and_callback, traverse_and_callback_mut},
    },
};

pub struct QueryPlanExecutionOpts<'exec> {
    pub query_plan: &'exec QueryPlan,
    pub projection_plan: &'exec [FieldProjectionPlan],
    pub headers_plan: &'exec HeaderRulesPlan,
    pub variable_values: &'exec Option<HashMap<String, sonic_rs::Value>>,
    pub extensions: HashMap<String, sonic_rs::Value>,
    pub client_request: &'exec ClientRequestDetails<'exec>,
    pub introspection_context: &'exec IntrospectionContext<'exec, 'static>,
    pub operation_type_name: &'exec str,
    pub executors: &'exec SubgraphExecutorMap,
    pub jwt_auth_forwarding: Option<JwtAuthForwardingPlan>,
    pub initial_errors: Vec<GraphQLError>,
    pub span: &'exec GraphQLOperationSpan,
}

#[derive(Default)]
pub struct PlanExecutionOutput {
    pub body: Vec<u8>,
    pub response_headers_aggregator: Option<ResponseHeaderAggregator>,
    pub error_count: usize,
}

pub async fn execute_query_plan<'exec>(
    opts: QueryPlanExecutionOpts<'exec>,
) -> Result<PlanExecutionOutput, PlanExecutionError> {
    let data = if let Some(introspection_query) = opts.introspection_context.query {
        resolve_introspection(introspection_query, opts.introspection_context)
    } else if opts.projection_plan.is_empty() {
        Value::Null
    } else {
        Value::Object(Vec::new())
    };

    let errors = opts.initial_errors;

    let extensions = opts.extensions;

    let query_plan = opts.query_plan;

    let dedupe_subgraph_requests = opts.operation_type_name == "Query";

    let mut exec_ctx = ExecutionContext::new(query_plan, data, errors);
    // No need for `new`, it has too many parameters
    // We can directly create `Executor` instance here
    let executor = Executor {
        variable_values: opts.variable_values,
        schema_metadata: opts.introspection_context.metadata,
        executors: opts.executors,
        client_request: opts.client_request,
        headers_plan: opts.headers_plan,
        jwt_forwarding_plan: opts.jwt_auth_forwarding,
        dedupe_subgraph_requests,
    };

    if let Some(node) = &query_plan.node {
        executor.execute_plan_node(&mut exec_ctx, node).await;
    }

    let error_count = exec_ctx.errors.len(); // Added for usage reporting

    let data = exec_ctx.data;
    let errors = exec_ctx.errors;
    let response_size_estimate = exec_ctx.response_storage.estimate_final_response_size();

    if error_count > 0 {
        opts.span.record_error_count(error_count);
        opts.span
            .record_errors(|| errors.iter().map(|e| e.into()).collect());
    }

    let body = project_by_operation(
        &data,
        errors,
        &extensions,
        opts.operation_type_name,
        opts.projection_plan,
        opts.variable_values,
        response_size_estimate,
        opts.introspection_context.metadata,
    )
    .with_plan_context(LazyPlanContext {
        subgraph_name: || None,
        affected_path: || None,
    })?;

    Ok(PlanExecutionOutput {
        body,
        response_headers_aggregator: exec_ctx.response_headers_aggregator.none_if_empty(),
        error_count,
    })
}

pub struct Executor<'exec> {
    variable_values: &'exec Option<HashMap<String, sonic_rs::Value>>,
    schema_metadata: &'exec SchemaMetadata,
    executors: &'exec SubgraphExecutorMap,
    client_request: &'exec ClientRequestDetails<'exec>,
    headers_plan: &'exec HeaderRulesPlan,
    jwt_forwarding_plan: Option<JwtAuthForwardingPlan>,
    dedupe_subgraph_requests: bool,
}

enum ExecutionJob<'exec> {
    Fetch {
        fetch_node_id: i64,
        subgraph_name: &'exec str,
        response: SubgraphResponse<'exec>,
    },
    FlattenFetch {
        fetch_node_id: i64,
        subgraph_name: &'exec str,
        response: SubgraphResponse<'exec>,
        flatten_node_path: &'exec FlattenNodePath,
        representation_hashes: Vec<u64>,
        representation_hash_to_index: HashMap<u64, usize>,
    },
}

impl<'exec> ExecutionJob<'exec> {
    fn response(self) -> SubgraphResponse<'exec> {
        match self {
            ExecutionJob::Fetch { response, .. } => response,
            ExecutionJob::FlattenFetch { response, .. } => response,
        }
    }
    fn response_ref(&self) -> &SubgraphResponse<'exec> {
        match self {
            ExecutionJob::Fetch { response, .. } => response,
            ExecutionJob::FlattenFetch { response, .. } => response,
        }
    }
    fn fetch_node_id(&self) -> i64 {
        match self {
            ExecutionJob::Fetch { fetch_node_id, .. } => *fetch_node_id,
            ExecutionJob::FlattenFetch { fetch_node_id, .. } => *fetch_node_id,
        }
    }
    fn subgraph_name(&self) -> &'exec str {
        match self {
            ExecutionJob::Fetch { subgraph_name, .. } => subgraph_name,
            ExecutionJob::FlattenFetch { subgraph_name, .. } => subgraph_name,
        }
    }
    fn affected_path(&self) -> Option<&'exec FlattenNodePath> {
        match self {
            ExecutionJob::Fetch { .. } => None,
            ExecutionJob::FlattenFetch {
                flatten_node_path, ..
            } => Some(flatten_node_path),
        }
    }
}

/// A unique identifier for a DAG node
type DagNodeId = usize;

/// Represents a node in the execution DAG
struct DagNode<'exec> {
    /// The original plan node to execute
    plan_node: &'exec PlanNode,
    /// Number of dependencies that must complete before this node can run
    dependency_count: usize,
    /// Nodes that depend on this node (to notify when this completes)
    dependents: Vec<DagNodeId>,
}

/// DAG scheduler for parallel execution
struct DagScheduler<'exec> {
    /// All nodes in the DAG
    nodes: Vec<DagNode<'exec>>,
    /// Current dependency counts (decremented as dependencies complete)
    remaining_deps: Vec<usize>,
    /// Queue of nodes ready to execute (have zero dependencies)
    ready_queue: VecDeque<DagNodeId>,
}

impl<'exec> DagScheduler<'exec> {
    /// Compile a PlanNode tree into a DAG
    fn compile(root: &'exec PlanNode) -> Self {
        let mut nodes = Vec::new();
        let mut node_id_counter = 0;
        
        // Build DAG from plan tree
        Self::compile_node(root, &mut nodes, &mut node_id_counter, None);
        
        // Initialize remaining dependency counts
        let remaining_deps = nodes.iter().map(|n| n.dependency_count).collect();
        
        // Initialize ready queue with ALL nodes that have no dependencies
        let mut ready_queue = VecDeque::new();
        for (node_id, node) in nodes.iter().enumerate() {
            if node.dependency_count == 0 {
                ready_queue.push_back(node_id);
            }
        }
        
        Self {
            nodes,
            remaining_deps,
            ready_queue,
        }
    }
    
    /// Recursively compile a plan node into DAG nodes
    /// Returns the node ID of the last created node, None if no nodes were created
    fn compile_node(
        node: &'exec PlanNode,
        nodes: &mut Vec<DagNode<'exec>>,
        id_counter: &mut usize,
        parent_id: Option<DagNodeId>,
    ) -> Option<DagNodeId> {
        match node {
            PlanNode::Sequence(seq) => {
                // For sequence: create chain where each node depends on previous
                let mut prev_id = parent_id;
                for child in &seq.nodes {
                    prev_id = Self::compile_node(child, nodes, id_counter, prev_id);
                }
                prev_id
            }
            PlanNode::Parallel(par) => {
                // For parallel: all children depend on parent (if any), but not on each other
                // Process all children with the same parent_id so they execute in parallel
                let mut last_id = None;
                for child in &par.nodes {
                    let child_id = Self::compile_node(child, nodes, id_counter, parent_id);
                    if child_id.is_some() {
                        last_id = child_id;
                    }
                }
                last_id
            }
            PlanNode::Condition(_cond) => {
                // Condition nodes need runtime evaluation - treat as a work node
                let node_id = *id_counter;
                *id_counter += 1;
                
                let dag_node = DagNode {
                    plan_node: node,
                    dependency_count: if parent_id.is_some() { 1 } else { 0 },
                    dependents: Vec::new(),
                };
                
                // Add dependency from parent if exists
                if let Some(pid) = parent_id {
                    if pid < nodes.len() {
                        nodes[pid].dependents.push(node_id);
                    }
                }
                
                nodes.push(dag_node);
                Some(node_id)
            }
            PlanNode::Fetch(_) | PlanNode::Flatten(_) => {
                // Leaf nodes that perform actual work
                let node_id = *id_counter;
                *id_counter += 1;
                
                let dag_node = DagNode {
                    plan_node: node,
                    dependency_count: if parent_id.is_some() { 1 } else { 0 },
                    dependents: Vec::new(),
                };
                
                // Add dependency from parent if exists
                if let Some(pid) = parent_id {
                    if pid < nodes.len() {
                        nodes[pid].dependents.push(node_id);
                    }
                }
                
                nodes.push(dag_node);
                Some(node_id)
            }
            _ => None, // Subscription, Defer nodes not yet supported in DAG
        }
    }
    
    /// Mark a node as completed and return newly ready nodes
    fn complete_node(&mut self, node_id: DagNodeId) -> Vec<DagNodeId> {
        let mut newly_ready = Vec::new();
        
        if node_id >= self.nodes.len() {
            return newly_ready;
        }
        
        // Notify all dependents
        for &dependent_id in &self.nodes[node_id].dependents {
            if dependent_id < self.remaining_deps.len() {
                self.remaining_deps[dependent_id] -= 1;
                
                // If dependent now has zero dependencies, it's ready
                if self.remaining_deps[dependent_id] == 0 {
                    newly_ready.push(dependent_id);
                }
            }
        }
        
        newly_ready
    }
}

impl<'exec> Executor<'exec> {
    async fn execute_plan_node(&self, ctx: &mut ExecutionContext<'exec>, node: &'exec PlanNode) {
        // Compile the plan tree into a DAG
        let mut scheduler = DagScheduler::compile(node);
        
        // Use FuturesUnordered for parallel execution
        let mut executing = FuturesUnordered::new();
        // Track which futures correspond to which node IDs
        let mut future_to_node: HashMap<usize, DagNodeId> = HashMap::new();
        let mut future_id_counter = 0;
        
        // Start executing ready nodes
        while let Some(ready_node_id) = scheduler.ready_queue.pop_front() {
            if let Some(fut) = self.prepare_node_future(&scheduler.nodes[ready_node_id].plan_node, &ctx.data) {
                let future_id = future_id_counter;
                future_id_counter += 1;
                future_to_node.insert(future_id, ready_node_id);
                executing.push(async move { (future_id, fut.await) }.boxed());
            } else {
                // Node had no work (e.g., Flatten with no data)
                // Complete it immediately and check for newly ready nodes
                let newly_ready = scheduler.complete_node(ready_node_id);
                scheduler.ready_queue.extend(newly_ready);
            }
        }
        
        // Process completions and launch newly ready nodes
        while let Some((future_id, job_result)) = executing.next().await {
            // Process the completed job
            self.process_job_result(ctx, job_result);
            
            // Mark node as complete and get newly ready nodes
            if let Some(&node_id) = future_to_node.get(&future_id) {
                let newly_ready = scheduler.complete_node(node_id);
                
                // Launch newly ready nodes
                for ready_node_id in newly_ready {
                    if let Some(fut) = self.prepare_node_future(&scheduler.nodes[ready_node_id].plan_node, &ctx.data) {
                        let new_future_id = future_id_counter;
                        future_id_counter += 1;
                        future_to_node.insert(new_future_id, ready_node_id);
                        executing.push(async move { (new_future_id, fut.await) }.boxed());
                    } else {
                        // Node had no work, complete it immediately
                        let more_ready = scheduler.complete_node(ready_node_id);
                        for rid in more_ready {
                            if let Some(fut) = self.prepare_node_future(&scheduler.nodes[rid].plan_node, &ctx.data) {
                                let new_future_id = future_id_counter;
                                future_id_counter += 1;
                                future_to_node.insert(new_future_id, rid);
                                executing.push(async move { (new_future_id, fut.await) }.boxed());
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Prepare a future for a single plan node (non-recursive)
    fn prepare_node_future<'wave>(
        &'wave self,
        node: &'exec PlanNode,
        data: &Value<'exec>,
    ) -> Option<BoxFuture<'wave, Result<ExecutionJob<'exec>, PlanExecutionError>>> {
        match node {
            PlanNode::Fetch(fetch_node) => {
                Some(self.prepare_fetch_job(fetch_node, None, None).boxed())
            }
            PlanNode::Flatten(flatten_node) => {
                self.prepare_flatten_job(flatten_node, data)
            }
            PlanNode::Condition(condition_node) => {
                // Evaluate condition and prepare the selected branch
                if let Some(selected_node) = condition_node_by_variables(condition_node, self.variable_values) {
                    self.prepare_node_future(selected_node, data)
                } else {
                    None
                }
            }
            // Sequence and Parallel should have been decomposed by compile_node
            _ => None,
        }
    }
    
    /// Prepare a job for a Flatten node
    fn prepare_flatten_job<'wave>(
        &'wave self,
        flatten_node: &'exec hive_router_query_planner::planner::plan_nodes::FlattenNode,
        data: &Value<'exec>,
    ) -> Option<BoxFuture<'wave, Result<ExecutionJob<'exec>, PlanExecutionError>>> {
        let fetch_node = match flatten_node.node.as_ref() {
            PlanNode::Fetch(fetch_node) => fetch_node,
            _ => return None,
        };
        let requires_nodes = fetch_node.requires.as_ref()?;

        let mut index = 0;
        let normalized_path = flatten_node.path.as_slice();
        let mut filtered_representations = Vec::new();
        filtered_representations.put(OPEN_BRACKET);
        let possible_types = &self.schema_metadata.possible_types;
        let mut representation_hashes: Vec<u64> = Vec::new();
        let mut representation_hash_to_index: HashMap<u64, usize> = HashMap::new();
        let arena = bumpalo::Bump::new();

        traverse_and_callback(data, normalized_path, self.schema_metadata, &mut |entity| {
            let hash = entity.to_hash(&requires_nodes.items, possible_types);

            if !entity.is_null() {
                representation_hashes.push(hash);
            }

            if representation_hash_to_index.contains_key(&hash) {
                return;
            }

            let entity = if let Some(input_rewrites) = &fetch_node.input_rewrites {
                let new_entity = arena.alloc(entity.clone());
                for input_rewrite in input_rewrites {
                    input_rewrite.rewrite(&self.schema_metadata.possible_types, new_entity);
                }
                new_entity
            } else {
                entity
            };

            let is_projected = project_requires(
                possible_types,
                &requires_nodes.items,
                entity,
                &mut filtered_representations,
                representation_hash_to_index.is_empty(),
                None,
            );

            if is_projected {
                representation_hash_to_index.insert(hash, index);
            }

            index += 1;
        });

        filtered_representations.put(CLOSE_BRACKET);

        if representation_hash_to_index.is_empty() {
            return None;
        }

        // This is the future for the actual fetch job
        Some(
            async {
                let fetch_job = self
                    .prepare_fetch_job(
                        fetch_node,
                        Some(filtered_representations),
                        Some(&flatten_node.path),
                    )
                    .await?;
                Ok(ExecutionJob::FlattenFetch {
                    flatten_node_path: &flatten_node.path,
                    response: fetch_job.response(),
                    fetch_node_id: fetch_node.id,
                    subgraph_name: fetch_node.service_name.as_str(),
                    representation_hashes,
                    representation_hash_to_index,
                })
            }
            .boxed(),
        )
    }

    // We handle `Result` instead of passing `PlanExecutionError` directly
    // as PipelineError so the first occurrence of an error does not stop the whole execution
    // But those errors are added to the final GraphQL response in `errors` field
    // of the GraphQL response
    // For example, if a subgraph is down, the rest of the plan can still be executed
    // See `error_handling_e2e_tests` for reproduction
    fn process_job_result(
        &self,
        ctx: &mut ExecutionContext<'exec>,
        job: Result<ExecutionJob<'exec>, PlanExecutionError>,
    ) {
        match job {
            Err(err) => {
                ctx.errors.push(err.into());
            }
            Ok(job) => {
                let subgraph_name = job.subgraph_name();
                let affected_path = job.affected_path();
                if let Some(ref subgraph_headers) = job.response_ref().headers {
                    if let Err(err) = apply_subgraph_response_headers(
                        self.headers_plan,
                        job.subgraph_name(),
                        subgraph_headers,
                        self.client_request,
                        &mut ctx.response_headers_aggregator,
                    )
                    .with_plan_context(LazyPlanContext {
                        subgraph_name: || Some(subgraph_name.to_string()),
                        affected_path: || affected_path.map(|p| p.to_string()),
                    }) {
                        ctx.errors.push(err.into());
                    }
                }

                let output_rewrites: Option<&[FetchRewrite]> =
                    ctx.output_rewrites.get(job.fetch_node_id());

                let (errors, entity_index_error_map) = match job {
                    ExecutionJob::Fetch { mut response, .. } => {
                        if let Some(response_bytes) = response.bytes {
                            ctx.response_storage.add_response(response_bytes);
                        }
                        if let Some(output_rewrites) = output_rewrites {
                            for output_rewrite in output_rewrites {
                                output_rewrite.rewrite(
                                    &self.schema_metadata.possible_types,
                                    &mut response.data,
                                );
                            }
                        }
                        deep_merge(&mut ctx.data, response.data);

                        (response.errors, None)
                    }
                    ExecutionJob::FlattenFetch {
                        mut response,
                        flatten_node_path,
                        representation_hashes,
                        ref representation_hash_to_index,
                        ..
                    } => {
                        if let Some(response_bytes) = response.bytes {
                            ctx.response_storage.add_response(response_bytes);
                        }
                        if let Some(mut entities) = response.data.take_entities() {
                            if let Some(output_rewrites) = output_rewrites {
                                for output_rewrite in output_rewrites {
                                    for entity in &mut entities {
                                        output_rewrite
                                            .rewrite(&self.schema_metadata.possible_types, entity);
                                    }
                                }
                            }

                            let mut index = 0;
                            let normalized_path = flatten_node_path.as_slice();
                            // If there is an error in the response, then collect the paths for normalizing the error
                            let initial_error_path = response.errors.as_ref().map(|_| {
                                GraphQLErrorPath::with_capacity(normalized_path.len() + 2)
                            });
                            let mut entity_index_error_map = response
                                .errors
                                .as_ref()
                                .map(|_| HashMap::with_capacity(entities.len()));
                            traverse_and_callback_mut(
                                &mut ctx.data,
                                normalized_path,
                                self.schema_metadata,
                                initial_error_path,
                                &mut |target, error_path| {
                                    let hash = representation_hashes[index];
                                    if let Some(entity_index) =
                                        representation_hash_to_index.get(&hash)
                                    {
                                        if let (Some(error_path), Some(entity_index_error_map)) =
                                            (error_path, entity_index_error_map.as_mut())
                                        {
                                            let error_paths = entity_index_error_map
                                                .entry(entity_index)
                                                .or_insert_with(Vec::new);
                                            error_paths.push(error_path);
                                        }
                                        if let Some(entity) = entities.get(*entity_index) {
                                            // SAFETY: `new_val` is a clone of an entity that lives for `'a`.
                                            // The transmute is to satisfy the compiler, but the lifetime
                                            // is valid.
                                            let new_val: Value<'_> =
                                                unsafe { std::mem::transmute(entity.clone()) };
                                            deep_merge(target, new_val);
                                        }
                                    }
                                    index += 1;
                                },
                            );
                            (response.errors, entity_index_error_map)
                        } else {
                            (response.errors, None)
                        }
                    }
                };

                ctx.handle_errors(subgraph_name, affected_path, errors, entity_index_error_map);
            }
        }
    }

    async fn prepare_fetch_job(
        &self,
        node: &'exec FetchNode,
        // If the fetch job is for a flatten node, we pass the filtered representations,
        representations: Option<Vec<u8>>,
        // and the path to the representations in the original response for error handling and normalization
        affected_path: Option<&FlattenNodePath>,
    ) -> Result<ExecutionJob<'exec>, PlanExecutionError> {
        let subgraph_operation_span = GraphQLSubgraphOperationSpan::new(
            node.service_name.as_str(),
            &node.operation.document_str,
        );

        async {
            // TODO: We could optimize header map creation by caching them per service name
            let mut headers_map = HeaderMap::new();
            let subgraph_name_factory = || Some(node.service_name.clone());
            let affected_path_factory = || affected_path.map(|p| p.to_string());
            modify_subgraph_request_headers(
                self.headers_plan,
                &node.service_name,
                self.client_request,
                &mut headers_map,
            )
            .with_plan_context(LazyPlanContext {
                subgraph_name: subgraph_name_factory,
                affected_path: affected_path_factory,
            })?;
            let variable_refs =
                select_fetch_variables(self.variable_values, node.variable_usages.as_ref());

            let mut subgraph_request = SubgraphExecutionRequest {
                query: node.operation.document_str.as_str(),
                dedupe: self.dedupe_subgraph_requests,
                operation_name: node.operation_name.as_deref(),
                variables: variable_refs,
                representations,
                headers: headers_map,
                extensions: None,
            };

            subgraph_operation_span.record_operation_identity(GraphQLSpanOperationIdentity {
                name: subgraph_request.operation_name,
                operation_type: match node.operation_kind {
                    Some(OperationKind::Query) | None => "query",
                    Some(OperationKind::Mutation) => "mutation",
                    Some(OperationKind::Subscription) => "subscription",
                },
                client_document_hash: node.operation.hash.to_string().as_str(),
            });

            if let Some(jwt_forwarding_plan) = &self.jwt_forwarding_plan {
                subgraph_request.add_request_extensions_field(
                    jwt_forwarding_plan.extension_field_name.clone(),
                    jwt_forwarding_plan.extension_field_value.clone(),
                );
            }

            let response = self
                .executors
                .execute(&node.service_name, subgraph_request, self.client_request)
                .await
                .with_plan_context(LazyPlanContext {
                    subgraph_name: subgraph_name_factory,
                    affected_path: affected_path_factory,
                })?;

            if let Some(errors) = &response.errors {
                if !errors.is_empty() {
                    subgraph_operation_span.record_error_count(errors.len());
                    subgraph_operation_span
                        .record_errors(|| errors.iter().map(|e| e.into()).collect());
                }
            }

            Ok(ExecutionJob::Fetch {
                fetch_node_id: node.id,
                subgraph_name: &node.service_name,
                response,
            })
        }
        .instrument(subgraph_operation_span.clone())
        .await
    }
}

fn condition_node_by_variables<'a>(
    condition_node: &'a ConditionNode,
    variable_values: &'a Option<HashMap<String, sonic_rs::Value>>,
) -> Option<&'a PlanNode> {
    let vars = variable_values.as_ref()?;
    let value = vars.get(&condition_node.condition)?;
    let condition_met = matches!(value.as_ref(), ValueRef::Bool(true));

    if condition_met {
        condition_node.if_clause.as_deref()
    } else {
        condition_node.else_clause.as_deref()
    }
}

fn select_fetch_variables<'a>(
    variable_values: &'a Option<HashMap<String, sonic_rs::Value>>,
    variable_usages: Option<&BTreeSet<String>>,
) -> Option<HashMap<&'a str, &'a sonic_rs::Value>> {
    let values = variable_values.as_ref()?;

    variable_usages.map(|variable_usages| {
        variable_usages
            .iter()
            .filter_map(|var_name| {
                values
                    .get_key_value(var_name.as_str())
                    .map(|(key, value)| (key.as_str(), value))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        context::ExecutionContext,
        execution::{
            client_request_details::{ClientRequestDetails, JwtRequestDetails, OperationDetails},
            plan::Executor,
        },
        headers::plan::HeaderRulesPlan,
        introspection::schema::SchemaMetadata,
        response::graphql_error::{GraphQLErrorExtensions, GraphQLErrorPath},
        SubgraphExecutorMap,
    };

    use super::select_fetch_variables;
    use hive_router_config::HiveRouterConfig;
    use hive_router_internal::telemetry::TelemetryContext;
    use hive_router_query_planner::{
        ast::{
            document::Document,
            operation::{OperationDefinition, SubgraphFetchOperation},
            selection_set::SelectionSet,
        },
        planner::plan_nodes::{FetchNode, ParallelNode, PlanNode},
    };
    use ntex::http::HeaderMap;
    use sonic_rs::Value;
    use std::{
        collections::{BTreeSet, HashMap},
        sync::{mpsc::channel, Arc},
        time::Duration,
        vec,
    };

    fn value_from_number(n: i32) -> Value {
        sonic_rs::from_str(&n.to_string()).unwrap()
    }

    #[test]
    fn select_fetch_variables_only_used_variables() {
        let mut variable_values_map = HashMap::new();
        variable_values_map.insert("used".to_string(), value_from_number(1));
        variable_values_map.insert("unused".to_string(), value_from_number(2));
        let variable_values = Some(variable_values_map);

        let mut usages = BTreeSet::new();
        usages.insert("used".to_string());

        let selected = select_fetch_variables(&variable_values, Some(&usages)).unwrap();

        assert_eq!(selected.len(), 1);
        assert!(selected.contains_key("used"));
        assert!(!selected.contains_key("unused"));
    }

    #[test]
    fn select_fetch_variables_ignores_missing_usage_entries() {
        let mut variable_values_map = HashMap::new();
        variable_values_map.insert("present".to_string(), value_from_number(3));
        let variable_values = Some(variable_values_map);

        let mut usages = BTreeSet::new();
        usages.insert("present".to_string());
        usages.insert("missing".to_string());

        let selected = select_fetch_variables(&variable_values, Some(&usages)).unwrap();

        assert_eq!(selected.len(), 1);
        assert!(selected.contains_key("present"));
        assert!(!selected.contains_key("missing"));
    }

    #[test]
    fn select_fetch_variables_for_no_usage_entries() {
        let mut variable_values_map = HashMap::new();
        variable_values_map.insert("unused_1".to_string(), value_from_number(1));
        variable_values_map.insert("unused_2".to_string(), value_from_number(2));

        let variable_values = Some(variable_values_map);

        let selected = select_fetch_variables(&variable_values, None);

        assert!(selected.is_none());
    }
    #[test]
    /**
     * We have the same entity in two different paths ["a", 0] and ["b", 1],
     * and the subgraph response has an error for this entity.
     * So we should duplicate the error for both paths.
     */
    fn normalize_entity_errors_correctly() {
        use crate::response::graphql_error::{GraphQLError, GraphQLErrorPathSegment};
        use std::collections::HashMap;
        let mut ctx = ExecutionContext::default();
        let mut entity_index_error_map: HashMap<&usize, Vec<GraphQLErrorPath>> = HashMap::new();
        entity_index_error_map.insert(
            &0,
            vec![
                GraphQLErrorPath {
                    segments: vec![
                        GraphQLErrorPathSegment::String("a".to_string()),
                        GraphQLErrorPathSegment::Index(0),
                    ],
                },
                GraphQLErrorPath {
                    segments: vec![
                        GraphQLErrorPathSegment::String("b".to_string()),
                        GraphQLErrorPathSegment::Index(1),
                    ],
                },
            ],
        );
        let response_errors = vec![GraphQLError {
            message: "Error 1".to_string(),
            locations: None,
            path: Some(GraphQLErrorPath {
                segments: vec![
                    GraphQLErrorPathSegment::String("_entities".to_string()),
                    GraphQLErrorPathSegment::Index(0),
                    GraphQLErrorPathSegment::String("field1".to_string()),
                ],
            }),
            extensions: GraphQLErrorExtensions::default(),
        }];
        ctx.handle_errors(
            "subgraph_a",
            None,
            Some(response_errors),
            Some(entity_index_error_map),
        );
        assert_eq!(ctx.errors.len(), 2);
        assert_eq!(ctx.errors[0].message, "Error 1");
        assert_eq!(
            ctx.errors[0].path.as_ref().unwrap().segments,
            vec![
                GraphQLErrorPathSegment::String("a".to_string()),
                GraphQLErrorPathSegment::Index(0),
                GraphQLErrorPathSegment::String("field1".to_string())
            ]
        );
        assert_eq!(ctx.errors[1].message, "Error 1");
        assert_eq!(
            ctx.errors[1].path.as_ref().unwrap().segments,
            vec![
                GraphQLErrorPathSegment::String("b".to_string()),
                GraphQLErrorPathSegment::Index(1),
                GraphQLErrorPathSegment::String("field1".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn runs_parallel_jobs_in_parallel() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let mut subgraph_a = mockito::Server::new_async().await;
        let mut subgraph_b = mockito::Server::new_async().await;
        let data = crate::response::value::Value::Null;
        let subgraph_endpoint_map = HashMap::from([
            (
                "subgraph_a".to_string(),
                format!("http://{}/graphql", subgraph_a.host_with_port())
                    .parse()
                    .unwrap(),
            ),
            (
                "subgraph_b".to_string(),
                format!("http://{}/graphql", subgraph_b.host_with_port())
                    .parse()
                    .unwrap(),
            ),
        ]);
        let executor = Executor {
            variable_values: &None,
            schema_metadata: &SchemaMetadata::default(),
            executors: &SubgraphExecutorMap::from_http_endpoint_map(
                &subgraph_endpoint_map,
                HiveRouterConfig::default().into(),
                Arc::new(TelemetryContext::from_propagation_config(
                    &Default::default(),
                )),
            )
            .unwrap(),
            client_request: &ClientRequestDetails {
                method: &http::Method::POST,
                url: &"http://example.com".parse().unwrap(),
                headers: &HeaderMap::new(),
                operation: OperationDetails {
                    name: None,
                    query: "{ from_a from_b }",
                    kind: "query",
                },
                jwt: JwtRequestDetails::Unauthenticated,
            },
            headers_plan: &HeaderRulesPlan::default(),
            jwt_forwarding_plan: None,
            dedupe_subgraph_requests: false,
        };

        let mock_a = subgraph_a
            .mock("POST", "/graphql")
            .with_body(r#"{"data":{"from_a":"value_a"}}"#)
            .create();

        let mut exec_ctx = ExecutionContext {
            data,
            ..Default::default()
        };

        // It is ok to have 'static lifetime here, because `data` is owned by `exec_ctx`, and `exec_ctx` lives for the entire duration of the test,
        // so the reference to `data` will never be dangling.
        let data_ref: &'static crate::response::value::Value<'static> =
            unsafe { std::mem::transmute(&exec_ctx.data) };

        let (sender, receiver) = channel();

        let mock_b = subgraph_b
            .mock("POST", "/graphql")
            .with_chunked_body(move |writer| {
                // We can add some delay here to make sure the parallel execution is actually working
                std::thread::sleep(Duration::from_millis(1000));
                // data should have `from_a` field from subgraph_a's response,
                // so data the merging process does not wait for subgraph_b's response to merge subgraph_a's response
                if let Some(data) = data_ref.as_object() {
                    let from_a_index = data.iter().position(|(k, _)| k == &"from_a");
                    let from_a_value = from_a_index
                        .and_then(|index| data.get(index))
                        .and_then(|(_, v)| v.as_str());
                    if let Some(from_a_value) = from_a_value {
                        sender
                            .send(from_a_value.to_string())
                            .expect("Failed to send from_a value through channel");
                    }
                }
                writer.write_fmt(format_args!(r#"{{"data":{{"from_b":"value_b"}}}}"#))
            })
            .create();

        let dummy_doc = Document {
            operation: OperationDefinition {
                name: None,
                operation_kind: None,
                variable_definitions: None,
                selection_set: SelectionSet { items: vec![] },
            },

            fragments: vec![],
        };

        executor
            .execute_plan_node(
                &mut exec_ctx,
                &PlanNode::Parallel(ParallelNode {
                    nodes: vec![
                        PlanNode::Fetch(FetchNode {
                            id: 1,
                            service_name: "subgraph_a".to_string(),
                            operation: SubgraphFetchOperation {
                                document_str: "{ from_a }".to_string(),
                                document: dummy_doc.clone(),
                                hash: 0,
                            },
                            operation_name: None,
                            requires: None,
                            input_rewrites: None,
                            output_rewrites: None,
                            variable_usages: None,
                            operation_kind: None,
                        }),
                        PlanNode::Fetch(FetchNode {
                            id: 2,
                            service_name: "subgraph_b".to_string(),
                            operation: SubgraphFetchOperation {
                                document_str: "{ from_b }".to_string(),
                                document: dummy_doc.clone(),
                                hash: 0,
                            },
                            operation_name: None,
                            requires: None,
                            input_rewrites: None,
                            output_rewrites: None,
                            variable_usages: None,
                            operation_kind: None,
                        }),
                    ],
                }),
            )
            .await;
        mock_a.assert();
        mock_b.assert();

        let from_a_value = receiver
            .recv()
            .expect("Failed to receive from_a value through channel");
        assert_eq!(from_a_value, "value_a");
    }

    #[tokio::test]
    async fn dag_scheduler_maintains_sequence_order() {
        // This test verifies that the DAG scheduler correctly maintains sequential
        // execution order when nodes are in a Sequence
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let mut subgraph_a = mockito::Server::new_async().await;
        let mut subgraph_b = mockito::Server::new_async().await;
        
        let data = crate::response::value::Value::Null;
        let subgraph_endpoint_map = HashMap::from([
            (
                "subgraph_a".to_string(),
                format!("http://{}/graphql", subgraph_a.host_with_port())
                    .parse()
                    .unwrap(),
            ),
            (
                "subgraph_b".to_string(),
                format!("http://{}/graphql", subgraph_b.host_with_port())
                    .parse()
                    .unwrap(),
            ),
        ]);
        
        let executor = Executor {
            variable_values: &None,
            schema_metadata: &SchemaMetadata::default(),
            executors: &SubgraphExecutorMap::from_http_endpoint_map(
                &subgraph_endpoint_map,
                HiveRouterConfig::default().into(),
                Arc::new(TelemetryContext::from_propagation_config(
                    &Default::default(),
                )),
            )
            .unwrap(),
            client_request: &ClientRequestDetails {
                method: &http::Method::POST,
                url: &"http://example.com".parse().unwrap(),
                headers: &HeaderMap::new(),
                operation: OperationDetails {
                    name: None,
                    query: "{ from_a from_b }",
                    kind: "query",
                },
                jwt: JwtRequestDetails::Unauthenticated,
            },
            headers_plan: &HeaderRulesPlan::default(),
            jwt_forwarding_plan: None,
            dedupe_subgraph_requests: false,
        };

        // Mock subgraphs - b should only be called after a completes
        let (sender, receiver) = channel();
        
        let mock_a = subgraph_a
            .mock("POST", "/graphql")
            .with_chunked_body(move |writer| {
                sender.send("a_started").unwrap();
                std::thread::sleep(Duration::from_millis(100));
                writer.write_fmt(format_args!(r#"{{"data":{{"from_a":"value_a"}}}}"#))
            })
            .create();

        let (sender_b, receiver_b) = channel();
        let mock_b = subgraph_b
            .mock("POST", "/graphql")
            .with_chunked_body(move |writer| {
                sender_b.send("b_started").unwrap();
                writer.write_fmt(format_args!(r#"{{"data":{{"from_b":"value_b"}}}}"#))
            })
            .create();

        let mut exec_ctx = ExecutionContext {
            data,
            ..Default::default()
        };

        let dummy_doc = Document {
            operation: OperationDefinition {
                name: None,
                operation_kind: None,
                variable_definitions: None,
                selection_set: SelectionSet { items: vec![] },
            },
            fragments: vec![],
        };
        
        // Test Plan: Sequence with two Fetch nodes
        // DAG scheduler should maintain sequential order
        executor
            .execute_plan_node(
                &mut exec_ctx,
                &PlanNode::Sequence(hive_router_query_planner::planner::plan_nodes::SequenceNode {
                    nodes: vec![
                        PlanNode::Fetch(FetchNode {
                            id: 1,
                            service_name: "subgraph_a".to_string(),
                            operation: SubgraphFetchOperation {
                                document_str: "{ from_a }".to_string(),
                                document: dummy_doc.clone(),
                                hash: 0,
                            },
                            operation_name: None,
                            requires: None,
                            input_rewrites: None,
                            output_rewrites: None,
                            variable_usages: None,
                            operation_kind: None,
                        }),
                        PlanNode::Fetch(FetchNode {
                            id: 2,
                            service_name: "subgraph_b".to_string(),
                            operation: SubgraphFetchOperation {
                                document_str: "{ from_b }".to_string(),
                                document: dummy_doc.clone(),
                                hash: 0,
                            },
                            operation_name: None,
                            requires: None,
                            input_rewrites: None,
                            output_rewrites: None,
                            variable_usages: None,
                            operation_kind: None,
                        }),
                    ],
                }),
            )
            .await;
        
        mock_a.assert();
        mock_b.assert();
        
        // Verify sequential execution: a must start before b
        let a_msg = receiver.recv().expect("Should receive a_started");
        assert_eq!(a_msg, "a_started");
        
        let b_msg = receiver_b.recv().expect("Should receive b_started");
        assert_eq!(b_msg, "b_started");
    }
}
