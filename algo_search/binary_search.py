import jax
import jax.numpy as jnp
from jax import lax

def binary_search(sorted_arr, target):
    """
    Binary search in JAX. Returns index of target if found, else -1.
    """
    n = len(sorted_arr)  # Get the length of the input array

    # Initial state: (low index, high index, found index)
    init_state = (0, n - 1, -1)

    # Condition function for the while loop
    def cond_fun(state):
        low, high, found = state
        # Continue while low <= high and target not found
        return (low <= high) & (found == -1)

    # Body function for the while loop
    def body_fun(state):
        low, high, found = state
        mid = (low + high) // 2  # Calculate the middle index

        mid_val = sorted_arr[mid]  # Get the value at the middle index

        # Update state based on comparison between mid_val and target
        new_low, new_high, new_found = lax.cond(
                mid_val < target,
                lambda _: (mid + 1, high, -1),      # Search right half
                lambda _: lax.cond(
                        mid_val > target,
                        lambda _: (low, mid - 1, -1),   # Search left half
                        lambda _: (low, high, mid),     # Found target
                        operand=None
                ),
                operand=None
        )

        return new_low, new_high, new_found

    # Run the while loop until the condition is False
    _, _, found_index = lax.while_loop(cond_fun, body_fun, init_state)

    return found_index  # Return the found index or -1 if not found

if __name__ == "__main__":
    # Example usage
    sorted_arr = jnp.array([1, 3, 5, 7, 9, 11])
    target = 7

    # Works with jit!
    jit_search = jax.jit(binary_search)
    index = jit_search(sorted_arr, target)
    print("Index:", index)  # → Index: 3

    # Test not found
    index = jit_search(sorted_arr, 6)
    print("Index:", index)  # → Index: -1