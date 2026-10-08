Fusion means combining these two ordered opinions into one final order. We normally cannot add their raw scores: keyword scoring and vector similarity use incompatible scales.

A common fusion method is Reciprocal Rank Fusion (RRF) [a method that rewards results appearing high in multiple lists]:

```text
fused score = 1 / (60 + keyword rank) + 1 / (60 + semantic rank)
```

Using it:

- A: rank 1 + rank 2 → strongest combined support
- B: rank 3 + rank 1 → next
- C: rank 2 + rank 3 → last

So final rank becomes: **A → B → C**.

In normal engineering words: keyword and semantic search are two independent retrieval services. Fusion is the adapter that merges their recommendations into one candidate queue. It does not decide the final answer; it decides which passages deserve to be considered next.


### Why 60 in RRF ?

Think of `60` as a **dampener knob**.

RRF gives every result a small credit based on its position:

```text
credit = 1 / (60 + rank)
```

Why add 60? So that rank 1 does not overpower everything else.

With `60`:

```text
rank 1 → 1/61 = 0.01639
rank 2 → 1/62 = 0.01613
```

Rank 1 is better, but only slightly better than rank 2. This lets fusion favor a passage that is consistently good in both searches.

If the number were `0`:

```text
rank 1 → 1/1 = 1
rank 2 → 1/2 = 0.5
```

Now rank 1 is massively stronger. One search engine putting a passage first could dominate the combined result, even if the other search engine does not rate it well.

So:

- **Larger number, such as 60:** rewards agreement across keyword and semantic search.
- **Smaller number:** makes the very top ranks much more powerful.
- **60:** a common starting value, not a law. Test it against real questions and tune it if needed.

The simplest memory sentence: **60 stops one “first place” vote from shouting too loudly.**