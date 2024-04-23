module PositionMeasurements

using Integrals, ClassicalOrthogonalPolynomials, Tullio

export assemble_position_operators, transverse_basis, label2image, label2image!

using PrecompileTools: @setup_workload, @compile_workload

function assemble_position_operators(xs, ys, basis)
    operators = Matrix{Matrix{ComplexF32}}(undef, length(xs), length(ys))

    Δx = (xs[2] - xs[1]) / 2
    Δy = (ys[2] - ys[1]) / 2

    function integrand!(y, r, par)
        for k ∈ eachindex(basis), j ∈ eachindex(basis)
            y[j, k] = conj(basis[j](r, par)) * basis[k](r, par)
        end
    end

    prototype = zeros(ComplexF32, length(basis), length(basis))
    f = IntegralFunction(integrand!, prototype)

    Threads.@threads for n ∈ eachindex(ys)
        for m ∈ eachindex(xs)
            domain = [xs[m] - Δx, ys[n] - Δy], [xs[m] + Δx, ys[n] + Δy]
            prob = IntegralProblem(f, domain)
            operators[m, n] = solve(prob, HCubatureJL()).u
        end
    end

    operators
end

function hg(x, y, m, n)
    N = 1 / sqrt(2^(m + n) * factorial(m) * factorial(n) * π)
    N * hermiteh(m, x) * hermiteh(n, y) * exp(-(x^2 + y^2) / 2)
end

function hg(x, y, m, n, x₀, y₀, γx, γy)
    hg((x - x₀) / γx, (y - y₀) / γy, m, n)
end

function transverse_basis(order)
    [(r, par) -> hg(r[1], r[2], order - n, n) for n ∈ 0:order]
end

function trasnverse_basis(order, pars...)
    [(r, par) -> hg(r[1], r[2], order - n, n, pars...) for n ∈ 0:order]
end

function transverse_basis(xd, yd, xc, yc, order, angle)
    basis = Array{complex(eltype(xd))}(undef, length(xd), length(yd), 2, order + 1)

    @tullio basis[i, j, 1, k] = hg(xd[i], yd[j], order - k + 1, k - 1)
    @tullio basis[i, j, 2, k] = hg(xc[i], yc[j], order - k + 1, k - 1)

    for k ∈ 0:order
        basis[:, :, 2, k+1] .*= cis(k * angle)
    end

    basis
end

function label2image!(dest, ψ::AbstractVector, basis)
    @tullio dest[i, j, m] = basis[i, j, m, k] * ψ[k] |> abs2
end

function label2image(ψ::AbstractVector, r, angle)
    basis = transverse_basis(r, r, r, r, size(ψ, 1) - 1, angle)
    image = Array{Float32,3}(undef, length(r), length(r), 2)
    label2image!(image, ψ, basis)
    image
end

function label2image!(dest, ρ::AbstractMatrix, basis)
    @tullio dest[i, j, k] = ρ[m, n] * basis[i, j, k, m] * conj(basis[i, j, k, n]) |> real
end

function label2image(ρ::AbstractMatrix, r, angle)
    basis = transverse_basis(r, r, r, r, size(ρ, 1) - 1, angle)
    image = Array{Float32,3}(undef, length(r), length(r), 2)
    label2image!(image, ρ, basis)
    image
end

@setup_workload begin
    rs = LinRange(-3, 3, 16)
    @compile_workload begin
        basis = transverse_basis(1)
        direct_operators = assemble_position_operators(rs, rs, basis)
    end
end

@setup_workload begin
    ψ = [1.0f0 + im, 0]
    ρ = zeros(ComplexF32, 2, 2)
    ρ[1, 1] = 1 / √2
    ρ[2, 2] = 1 / √2
    rs = LinRange(-3, 3, 16)
    @compile_workload begin
        basis = transverse_basis(rs, rs, rs, rs, 1, π / 2)
        image = label2image(ψ, rs, π / 2)
        image = label2image(ρ, rs, π / 2)
    end
end

end
