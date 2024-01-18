using Combinatorics

function multinomial(a, list_b)
    inner_sum = BigInt(0)
    inner_prod = BigInt(1)
    if any(x->x<0, list_b)
        return BigInt(0)
    end
    for b in list_b
        inner_sum += b
        inner_prod *= binomial(BigInt(inner_sum), BigInt(b))
    end
    return inner_prod
end

function alpha_l1_l2_l3(n, l1, l2, l3)
    a = multinomial(n, [l1, l2, l3, n-l1-l2-l3])
    b = multinomial(2*n, [2*l1, 2*l2, 2*l3, 2*(n-l1-l2-l3)])
    c = binomial(BigInt(2*n), BigInt(2*(l1+l3)))
    d = binomial(BigInt(n), BigInt(l1+l3))
    e = binomial(BigInt(2*n), BigInt(2*(l2+l3)))
    f = binomial(BigInt(n), BigInt(l2+l3))
    return BigFloat(a)/BigFloat(b)*BigFloat(c)/BigFloat(d)*BigFloat(e)/BigFloat(f)
end

function kappa(n, zeta, l1, l2, l3)
    inner_sum(i, zeta) = binomial(BigInt(zeta), BigInt(2*i))*multinomial(n-zeta, [l1-zeta/2+i, l2-zeta/2+i, l3-i, n-l1-l2-l3-i])
    return BigFloat(2)^zeta * sum([inner_sum(i, zeta) for i in 0:zeta//2])
end

function b(n, zeta)
    inner_sum = zeros(Float64, n+1, n+1, n+1)
    # Threads.@threads 
    for l1 in 0:n
        for l2 in 0:n
            for l3 in 0:n
                if l1+l2+l3 <= n
                    inner_sum[l1+1, l2+1, l3+1] = alpha_l1_l2_l3(n, l1, l2, l3) * kappa(n, zeta, l1, l2, l3)
                end
            end
        end
    end
    return 0.5^(2*n)*sum(inner_sum)
end


@. power_law(x, params) = params[1] * x[:, 1]^params[2] * x[:, 2]^params[3]
@. line(x, params) = params[1] + x[:, 1] * params[2] + x[:, 2] * params[3]


nacts = collect(2:4:60)
ret = []
Threads.@threads for n in nacts
    # for nel in 2:2:2*n
    #     println(n, " ", nel)
    #     bnd = b(2*n, nel)
    #     push!(ret, [2*n nel bnd])
    # end
    println(n)
    bnd = b(2*n, n)
    push!(ret, [2*n n bnd])
end
ret = reduce(vcat, ret)

xy = ret[:, 1:2]
bounds = ret[:, 3]

using LsqFit
using DataFrames
using CSV

@show size(bounds)
fit = curve_fit(power_law, xy, bounds, [1., 1., 1.])
@show fit

fit_lin = curve_fit(line, log.(xy), log.(bounds), [1., 1., 1.])
@show fit_lin


df = DataFrame(ret, ["n", "nel", "bound"])
CSV.write("bound.csv", df)

# @time begin
#     bound = b(8, 4)
#     println(bound)
# end

# @time begin
#     bound = b(100, 50)
#     println(bound)
# end