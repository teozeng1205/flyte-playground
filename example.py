import flyte

def my_func():
    ...

# A TaskEnvironment provides a way of grouping the configuration used by tasks.
env = flyte.TaskEnvironment(
    name="hello-world",
    # resources=flyte.Resources(cpu=1, memory="2Gi", gpu="A100:4"),
    image=(
        flyte.Image
        .from_debian_base((3, 12))
        .with_pip_packages("pandas",)
    ),
    env_vars={"SOME_ENV_VAR": "some_value"},
    secrets=[flyte.Secret("WANDB_API_KEY")],
    cache=flyte.Cache("auto"),
)

# Use a TaskEnvironment to define tasks, which are regular Python functions.
@env.task
def square(x: int) -> int:

    out = x * x

    return out

@env.task
def sqrt(x: float|int) -> float:
    return x ** 0.5

# Tasks can call other tasks.
# Each task defined with a given TaskEnvironment will run in its own separate container,
# but the containers will all be configured identically.
@env.task
def main(values: list[int]) -> int:

    ten_squared = square(10)

    square(ten_squared + 5)

    sqrt(16)

    try:
        sqrt(-1)
    except Exception as e:
        print(f"Caught an error: {e}")

    for value in [10, 20, 30]:
        print(f"The square of {value} is {square(value)}")

    return sum(list(flyte.map(square, values)))



if __name__ == "__main__":

    # Establish a remote connection from within your script.
    flyte.init_from_config(".flyte/config.yaml")

    # Run your tasks remotely inline and pass parameter data.
    run = flyte.run(main, values=list(range(10)))

    # Print various attributes of the run.
    print(run.name)
    print(run.url)

    # run.wait(run)