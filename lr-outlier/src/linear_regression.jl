# ....................
# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: julia
# @file: /try.jl
# @created: Thursday, 26th March 2020
# @author: brentian (chuwzhang@126.com)
# @modified: brentian (chuwzhang@126.com>)
#    Thursday, 26th March 2020 3:31:03 pm

# @description: 
__precompile__()
module lr_with_outlier

using JuMP
using Gurobi
using Plots
using LinearAlgebra
using Printf

# export module functions
export f

function f(x, y0, n, m, N, x_min, x_max, y_min, y_max)
    model = Model()
    # Declaring variables
    @variable(model, b[1:m] >= 0)
    # @variable(model, 0 <= q[1:n] <= 1)
    @variable(model, q[1:n], Bin)

    @variable(model, res[1:n])
    # == absolute value ==
    @constraint(model, RES[i = 1:n], res[i] + dot(b, x[i, :]) >= y0[i])
    @constraint(model, RES_NEG[i = 1:n], res[i] - dot(b, x[i, :]) >= -y0[i])
    # == quadratic ==
    # @constraint(model, 
    #   RES_QUAD[i = 1:n], 
    #   [res[i], dot(b, x[i,:]) - y0[i]] in SecondOrderCone())
    block_m = [
        1 * diagm(ones(n)) 0.5 * diagm(ones(n))
        0.5 * diagm(ones(n)) 1 * diagm(ones(n))
    ]
    @constraint(model, sum(q) >= N)
    @objective(model, Min, dot([res; q], block_m, [res; q]))
    set_optimizer(model, Gurobi.Optimizer)

    # Printing the prepared optimization model
    print(model)


    # Solving the optimization problem
    optimize!(model)

    print(value.(q), '\n')
    print(value.(b), '\n')

    # === summarizing results ===
    q1 = value.(q)

    # metrics
    y_bar = sum(y0) / length(y0)
    s_tot = sum((y0 .- y_bar).^2 .* q1)
    r = value.(res)
    s_res = sum(r.^2 .* q1)


    # make inferences 
    y_pred = [dot(value.(b), x[i, :]) * (y_max - y_min) + y_min for i in 1:n]

    selected_index = [i for i = 1:n if q1[i] == 1]
    dropped_index = [i for i = 1:n if q1[i] != 1]
    s_x = x[selected_index] .* (x_max - x_min) .+ x_min
    d_x = x[dropped_index] .* (x_max - x_min) .+ x_min
    s_y0 = y0[selected_index] .* (y_max - y_min) .+ y_min
    d_y0 = y0[dropped_index] .* (y_max - y_min) .+ y_min
    s_y_pred = y_pred[selected_index]
    d_y_pred = y_pred[dropped_index]
    r_sqr = (1 - s_res / s_tot)
    p = plot(
        s_x,
        s_y_pred,
        lab = ["selected_regression"],
        title = "drop $(n - N) =>\nR^2: $r_sqr",
        legend = false,
    )
    scatter!(s_x, s_y0, lab = ["selected_samples"])
    scatter!(d_x, d_y0, lab = ["outliers"])
    return [p, value.(b), N, q1, r_sqr, s_x, s_y_pred]
end

end
