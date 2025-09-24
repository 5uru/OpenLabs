import jax
import jax.numpy as jnp
from jax import lax

def merge_sort(arr):
    # Helper function to merge two sorted arrays
    def merge(left, right):
        left_len = left.shape[0]
        right_len = right.shape[0]
        total_len = left_len + right_len

        # Condition function for the while loop: continue until merged array is filled
        def cond_fun(state):
            i, j, merged, k = state
            return k < total_len

        # Body function for the while loop: select the next smallest element from left or right
        def body_fun(state):
            i, j, merged, k = state
            take_left = (i < left_len) & ((j >= right_len) | (left[i] <= right[j]))
            new_val = jnp.where(take_left, left[i], right[j])
            new_i = i + take_left
            new_j = j + (~take_left)
            merged = merged.at[k].set(new_val)
            return new_i, new_j, merged, k + 1

        # Preallocate merged array
        merged = jnp.empty((total_len,), dtype=arr.dtype)
        # Run the while loop to merge left and right arrays
        _, _, merged, _ = lax.while_loop(cond_fun, body_fun, (0, 0, merged, 0))
        return merged

    # Recursive sort function
    def sort(sub_arr):
        # Base case: array of length 1 or less is already sorted
        if sub_arr.shape[0] <= 1:
            return sub_arr
        mid = sub_arr.shape[0] // 2
        # Recursively sort left and right halves
        left = sort(sub_arr[:mid])
        right = sort(sub_arr[mid:])
        # Merge sorted halves
        return merge(left, right)

    # Start the recursive sort
    return sort(arr)

if __name__ == "__main__":
    arr = jnp.array([64, 25, 12, 22, 11])
    jit_sort = jax.jit(merge_sort)
    sorted_arr = jit_sort(arr)
    print("Sorted array:", sorted_arr)
