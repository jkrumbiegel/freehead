using LinearAlgebra
using Statistics

struct Rigidbody
    markers::Array{Float64}
    ref_points::Array{Float64}
end

function solve(r::Rigidbody, multiple_markers::Array{Float64, 3})

    n_frames = size(multiple_markers)[3]
    rotations = fill(NaN, (3, 3, n_frames))
    ref_points = fill(NaN, (size(r.ref_points)..., n_frames))

    for i in 1:n_frames
        rotations[:, :, i], ref_points[:, :, i] = solve(r, multiple_markers[:, :, i])
    end

    return rotations, ref_points

end

function solve(r::Rigidbody, single_markers::Array{Float64, 2})
    n_markers = size(single_markers)
    valid_markers_mask = vec(all(
        isfinite.(single_markers),
        dims=1
    ))
    n_valid = sum(valid_markers_mask)

    if n_valid < 3
        return fill(NaN, (3, 3))
    end

    valid_markers = single_markers[:, valid_markers_mask]
    reference_subset = r.markers[:, valid_markers_mask]

    valid_markers_centroid = mean(valid_markers, dims=2)
    valid_markers_centered = valid_markers .- valid_markers_centroid

    reference_subset_centroid = mean(reference_subset, dims=2)
    reference_subset_centered = reference_subset .- reference_subset_centroid

    ref_center_to_ref_points = r.ref_points .- reference_subset_centroid

    rotation = orthogonal_procrustes(valid_markers_centered, reference_subset_centered)

    ref_points_translated = rotation * ref_center_to_ref_points .+ valid_markers_centroid

    return rotation, ref_points_translated

end

function orthogonal_procrustes(measured, reference)

    F = svd(measured * reference')

    R = F.V * F.U'
    if abs(det(R) + 1) < 1e-10
        R = F.V * diagm(0 => [1.0, 1.0, -1.0]) * F.U'
    end

    return R

end
