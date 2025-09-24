import jax
import jax.numpy as jnp
from jax import lax

def insertion_sort(arr):
    # Get the length of the input array
    n = arr.shape[0]

    # Condition function for the outer while loop (controls the main insertion sort loop)
    def outer_cond(state):
        idx, _ = state
        return idx < n

    # Body function for the outer while loop
    def outer_body(state):
        idx, arr = state
        key = arr[idx]  # The element to insert
        j = idx - 1     # Start comparing from the previous element

        # Condition for the inner while loop (shifts elements greater than key to the right)
        def inner_cond(inner_state):
            j, arr = inner_state
            return (j >= 0) & (arr[j] > key)

        # Body for the inner while loop (performs the shift)
        def inner_body(inner_state):
            j, arr = inner_state
            arr = arr.at[j + 1].set(arr[j])  # Shift element right
            return j - 1, arr

        # Run the inner while loop to shift elements
        j, arr = lax.while_loop(inner_cond, inner_body, (j, arr))
        arr = arr.at[j + 1].set(key)  # Insert the key in the correct position
        return idx + 1, arr

    # Run the outer while loop to sort the entire array
    _, sorted_arr = lax.while_loop(outer_cond, outer_body, (1, arr))
    return sorted_arr

if __name__ == "__main__":
    arr = jnp.array([64, 25, 12, 22, 11])
    jit_sort = jax.jit(insertion_sort)
    sorted_arr = jit_sort(arr)
    print("Sorted array:", sorted_arr)
