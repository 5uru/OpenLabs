import jax
import jax.numpy as jnp
from jax import lax

def bubble_sort(arr):
    n = arr.shape[0]

    # Outer loop condition: continue if not at end and a swap occurred last pass
    def cond_fun(state):
        arr, i, swapped = state
        return jnp.logical_and(i < n - 1, swapped)

    # Outer loop body: perform one pass of bubble sort
    def body_fun(state):
        arr, i, _ = state
        swapped = False

        # Inner loop body: compare and swap adjacent elements if needed
        def inner_body_fun(inner_state):
            arr, j, swapped = inner_state

            # Swap arr[j] and arr[j+1] if arr[j] > arr[j+1]
            def swap_fn(a):
                a = a.at[j].set(arr[j + 1])
                a = a.at[j + 1].set(arr[j])
                return a

            should_swap = arr[j] > arr[j + 1]
            arr_new = lax.cond(should_swap, swap_fn, lambda a: a, arr)
            swapped_new = jnp.logical_or(swapped, should_swap)
            return arr_new, j + 1, swapped_new

        # Inner loop condition: iterate through unsorted part of array
        def inner_cond_fun(inner_state):
            _, j, _ = inner_state
            return j < n - i - 1

        # Run inner loop for one pass
        arr, _, swapped = lax.while_loop(
                inner_cond_fun, inner_body_fun, (arr, 0, swapped)
        )
        return arr, i + 1, swapped

    # Initial state: (array, outer index, swapped flag)
    init_state = (arr, 0, True)
    # Run outer loop until sorted
    sorted_arr, _, _ = lax.while_loop(cond_fun, body_fun, init_state)
    return sorted_arr

if __name__ == "__main__":
    arr = jnp.array([64, 34, 25, 12, 22, 11, 90])
    jit_sort = jax.jit(bubble_sort)
    sorted_arr = jit_sort(arr)
    print("Sorted array:", sorted_arr)
