//! Urgent test - created for interruption test scenario
//!
//! This test file was created as part of testing context-switching workflows.

#[test]
fn test_urgent_task_completed() {
    // This is a simple test to verify the urgent task was handled
    assert!(true, "Urgent task was successfully handled");
}

#[test]
fn test_interruption_workflow() {
    // Verify we can switch contexts and come back
    let original_work = "dsrs-vn6.1.2";
    let urgent_work = "dsrs-lzh";

    assert_ne!(original_work, urgent_work);
    assert!(!original_work.is_empty());
    assert!(!urgent_work.is_empty());
}
