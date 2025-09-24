import jax
import jax.numpy as jnp
from jax import lax

def quick_sort(arr):
    """
    JIT‑compatible iterative quicksort for 1D arrays.
    """
    # Validate input shape early (static at trace time).
    if arr.ndim != 1:
        raise ValueError("jit_quicksort expects a 1D array")
    if arr.shape[0] <= 1:
        return arr

    n = arr.shape[0]
    max_stack = n  # Worst‑case depth for quicksort

    # Allocate fixed‑size stack of index pairs [low, high] (int32 for JAX indexing).
    stack = jnp.zeros((max_stack, 2), dtype=jnp.int32)
    # Seed the stack with the initial full range [0, n‑1].
    stack = stack.at[0].set(jnp.array([0, n - 1], dtype=jnp.int32))

    # State tuple carried by the outer while loop: (array, stack, stack_ptr).
    state = (arr, stack, jnp.array(1, dtype=jnp.int32))

    def swap(a, i, j):
        """Swap elements a[i] and a[j] using indexed updates (JIT‑safe)."""
        vi = a[i]
        vj = a[j]
        a = a.at[i].set(vj)
        a = a.at[j].set(vi)
        return a

    def partition(a, low, high):
        """
        Lomuto partition around pivot a[high].

        Walk j from low to high‑1.
        Maintain i as the index of the last element <= pivot.
        When a[j] <= pivot, advance i and swap a[i] with a[j].
        Finally swap pivot into position i+1 and return that index.
        """
        pivot = a[high]
        i = low - jnp.int32(1)
        j = low

        def cond_fn(loop_state):
            # Continue while j < high.
            _, _i, _j, _pivot, _high = loop_state
            return _j < _high

        def body_fn(loop_state):
            a, i, j, pivot, high = loop_state
            vj = a[j]
            le = vj <= pivot
            # If a[j] <= pivot, move boundary i forward.
            new_i = lax.select(le, i + jnp.int32(1), i)
            # Conditionally swap a[new_i] with a[j].
            a = lax.cond(le, lambda x: swap(x, new_i, j), lambda x: x, a)
            return (a, new_i, j + jnp.int32(1), pivot, high)

        # Loop j across the segment.
        a, i, _, _, _ = lax.while_loop(cond_fn, body_fn, (a, i, j, pivot, high))
        ip1 = i + jnp.int32(1)
        # Place pivot into its final position.
        a = swap(a, ip1, high)
        return a, ip1

    def process_segment(a, low, high):
        """
        If low < high, partition the segment and return (a, pivot_index).
        Otherwise return (a, low) as a no‑op pivot marker.
        """
        def tfn(ops):
            a, low, high = ops
            return partition(a, low, high)

        def ffn(ops):
            a, low, _ = ops
            return (a, low)

        return lax.cond(low < high, tfn, ffn, operand=(a, low, high))

    def cond(state):
        # Continue while there are pending segments on the stack.
        _, _, stack_ptr = state
        return stack_ptr > 0

    def body(state):
        a, stk, sptr = state

        # Pop one segment [low, high].
        sptr = sptr - jnp.int32(1)
        low = stk[sptr, 0]
        high = stk[sptr, 1]

        # Partition the current segment if needed.
        a, pidx = process_segment(a, low, high)

        # Push right segment [pidx+1, high] if it contains 2+ elements.
        def push_right(ops):
            sp, s, p, h = ops
            s = s.at[sp, 0].set(p + jnp.int32(1)).at[sp, 1].set(h)
            return (sp + jnp.int32(1), s)

        def skip_right(ops):
            sp, s, *_ = ops
            return (sp, s)

        sptr, stk = lax.cond(
                pidx + jnp.int32(1) <= high,
                push_right,
                skip_right,
                operand=(sptr, stk, pidx, high),
                )

        # Push left segment [low, pidx‑1] if it contains 2+ elements.
        def push_left(ops):
            sp, s, l, p = ops
            s = s.at[sp, 0].set(l).at[sp, 1].set(p - jnp.int32(1))
            return (sp + jnp.int32(1), s)

        def skip_left(ops):
            sp, s, *_ = ops
            return (sp, s)

        sptr, stk = lax.cond(
                low <= pidx - jnp.int32(1),
                push_left,
                skip_left,
                operand=(sptr, stk, low, pidx),
                )

        return (a, stk, sptr)

    # Main iterative quicksort loop over the explicit stack.
    sorted_arr, _, _ = lax.while_loop(cond, body, state)
    return sorted_arr

# JIT‑compile the function for performance.
sorted_fn = jax.jit(quick_sort)

if __name__ == "__main__":
    # Example usage.
    arr = jnp.array([3, 6, 8, 10, 1, 2, 1])
    out = sorted_fn(arr)
    print("Sorted array:", out)
