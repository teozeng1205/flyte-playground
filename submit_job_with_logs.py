import flyte

# Define environment with PYTHONUNBUFFERED=1 to enable log flushing
env = flyte.TaskEnvironment(
    name="job-with-logs",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=(
        flyte.Image
        .from_debian_base((3, 12))
        .with_pip_packages("pandas")
    ),
    env_vars={
        "PYTHONUNBUFFERED": "1",  # Enable unbuffered logs
    },
    cache=flyte.Cache("auto"),
)

@env.task
def process_data(n: int) -> dict:
    """Example task that prints logs - they will be flushed immediately."""
    import time

    print("Starting data processing...")

    for i in range(n):
        print(f"Processing item {i+1}/{n}")
        time.sleep(0.5)  # Simulate work

    print("Data processing complete!")

    return {
        "status": "completed",
        "items_processed": n,
    }

@env.task
def main(num_items: int = 10) -> dict:
    """Main workflow that processes data."""
    print(f"Starting workflow with {num_items} items")

    result = process_data(num_items)

    print(f"Workflow complete: {result}")
    return result


if __name__ == "__main__":
    # Initialize connection to Flyte backend
    flyte.init_from_config(".flyte/config.yaml")

    # Submit the job remotely
    run = flyte.run(main, num_items=10)

    # Print run information
    print(f"Job submitted successfully!")
    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
    print(f"\nVisit the URL above to view logs and monitor progress.")

    # Uncomment the line below to wait for completion and get results
    # result = run.wait()
    # print(f"Final result: {result}")
