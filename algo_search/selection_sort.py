import jax
import jax.numpy as jnp
from jax import lax

def selection_sort(arr):
    """
    Selection sort in JAX. Returns a sorted array.
    """
    n = len(arr)  # Get the length of the input array

    # Initial state: (current index, array)
    init_state = (0, arr)

    # Condition function for the while loop
    def cond_fun(state):
        idx, _ = state
        # Continue while idx < n - 1
        return idx < n - 1

    # Body function for the while loop
    def body_fun(state):
        idx, arr = state

        # Find the index of the minimum element in the unsorted portion
        def find_min(j, min_idx):
            return lax.cond(
                arr[j] < arr[min_idx],
                lambda _: j,
                lambda _: min_idx,
                operand=None
            )

        min_idx = lax.fori_loop(idx + 1, n, find_min, idx)

        # Swap the found minimum element with the first element of the unsorted portion
        arr = arr.at[idx].set(arr[min_idx])
        arr = arr.at[min_idx].set(arr[idx])

        return idx + 1, arr

    # Run the while loop until the condition is False
    _, sorted_arr = lax.while_loop(cond_fun, body_fun, init_state)

    return sorted_arr  # Return the sorted array
if __name__ == "__main__":
    # Example usage
    arr = jnp.array([64, 25, 12, 22, 11])

    # Works with jit!
    jit_sort = jax.jit(selection_sort)
    sorted_arr = jit_sort(arr)
    print("Sorted array:", sorted_arr)  # → Sorted array: [11 12 22 25 64]