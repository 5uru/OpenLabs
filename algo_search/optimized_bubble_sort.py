import jax
from jax import lax

def optimized_bubble_sort(arr):
    # Get the length of the array
    n = arr.shape[0]

    # Outer loop: iterate over each pass
    def body_fun(i, arr):
        # Inner loop: iterate through unsorted part of the array
        def inner_body_fun(j, arr):
            # Check if current element is greater than the next
            swap = arr[j] > arr[j + 1]
            # Conditionally swap elements using JAX's functional update
            arr = jax.lax.cond(
                    swap,
                    lambda x: x.at[j].set(arr[j + 1]).at[j + 1].set(arr[j]),
                    lambda x: x,
                    arr
            )
            return arr

        # Perform inner loop for current pass
        arr = lax.fori_loop(0, n - i - 1, inner_body_fun, arr)
        return arr

    # Perform outer loop for all passes
    arr = lax.fori_loop(0, n - 1, body_fun, arr)
    return arr


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    arr = jax.random.randint(key, (10,), 0, 100)
    print("Original array:", arr)
    sorted_arr = optimized_bubble_sort(arr)
    print("Sorted array:", sorted_arr)