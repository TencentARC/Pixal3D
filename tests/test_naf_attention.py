import torch

from backends.naf_attention import chunked_na2d, neighborhood_indices


def _brute_na2d(query, key, value, kernel_size, dilation, scale):
    batch, height, width, heads, head_dim = query.shape
    y_indices = neighborhood_indices(
        height, kernel_size, dilation, device=query.device
    )
    x_indices = neighborhood_indices(
        width, kernel_size, dilation, device=query.device
    )
    output = torch.empty(
        batch,
        height,
        width,
        heads,
        value.shape[-1],
        dtype=value.dtype,
        device=value.device,
    )
    for y in range(height):
        for x in range(width):
            keys = []
            values = []
            for ny in y_indices[y]:
                for nx in x_indices[x]:
                    keys.append(key[:, ny, nx])
                    values.append(value[:, ny, nx])
            keys = torch.stack(keys, dim=-2)
            values = torch.stack(values, dim=-2)
            logits = torch.einsum("bhd,bhkd->bhk", query[:, y, x], keys)
            weights = torch.softmax(logits * scale, dim=-1)
            output[:, y, x] = torch.einsum(
                "bhk,bhkd->bhd", weights, values
            )
    return output


def test_neighborhood_indices_shift_windows_at_boundaries():
    indices = neighborhood_indices(
        length=8,
        kernel_size=3,
        dilation=1,
        device=torch.device("cpu"),
    )
    assert indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [5, 6, 7],
    ]


def test_neighborhood_indices_preserve_dilation_groups():
    indices = neighborhood_indices(
        length=12,
        kernel_size=3,
        dilation=2,
        device=torch.device("cpu"),
    )
    for query, neighbors in enumerate(indices.tolist()):
        assert all(index % 2 == query % 2 for index in neighbors)


def test_chunked_na2d_matches_brute_reference():
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(1, 12, 10, 2, 4, generator=generator)
    key = torch.randn(1, 12, 10, 2, 4, generator=generator)
    value = torch.randn(1, 12, 10, 2, 7, generator=generator)
    scale = 4**-0.5
    expected = _brute_na2d(query, key, value, 3, 2, scale)

    for chunk_rows in (1, 3, 12):
        actual = chunked_na2d(
            query,
            key,
            value,
            kernel_size=3,
            dilation=2,
            scale=scale,
            chunk_rows=chunk_rows,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
