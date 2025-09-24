import jax
import jax.numpy as jnp
from jax import lax

def linear_search(arr, target):
    """
    Linear search in JAX. Returns index of target if found, else -1.
    """
    n = len(arr)  # Get the length of the input array

    # Initial state: (current index, found index)
    init_state = (0, -1)

    # Condition function for the while loop
    def cond_fun(state):
        idx, found = state
        # Continue while idx < n and target not found
        return (idx < n) & (found == -1)

    # Body function for the while loop
    def body_fun(state):
        idx, found = state
        current_val = arr[idx]  # Get the value at the current index

        # Update state based on comparison between current_val and target
        new_idx, new_found = lax.cond(
            current_val == target,
            lambda _: (idx + 1, idx),  # Found target
            lambda _: (idx + 1, -1),   # Continue searching
            operand=None
        )

        return new_idx, new_found

    # Run the while loop until the condition is False
    _, found_index = lax.while_loop(cond_fun, body_fun, init_state)

    return found_index  # Return the found index or -1 if not found

if __name__ == "__main__":
    # Example usage
    arr = jnp.array([4, 2, 7, 1, 3, 5])
    target = 3

    # Works with jit!
    jit_search = jax.jit(linear_search)
    index = jit_search(arr, target)
    print("Index:", index)  # → Index: 4

    # Test not found
    index = jit_search(arr, 6)
    print("Index:", index)  # → Index: -1