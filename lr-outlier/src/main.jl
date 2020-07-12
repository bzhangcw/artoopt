# ....................
# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: lr_with_outlier
# @file: /main.jl
# @created: Sunday, 5th April 2020
# @author: brentian (chuwzhang@126.com)
# @modified: brentian (chuwzhang@126.com>)
#    Sunday, 5th April 2020 4:14:00 pm
# ....................
# @description: 
# real data
module MyApp

include("./linear_regression.jl")
using .lr_with_outlier
using Plots
# define size := m, n

Base.@ccallable function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function real_main()
    # Preparing an optimization model
    # === 2-d
    # n, m = 10, 2
    # x = [ ones(n) [i * 1.0 for i = 1:n]]
    # y = Array(2:n + 1)
    # y[3] = 100 
    # === 1-d
    # n, m = 10, 1
    # x =  [i * 1.0 for i = 1:n]
    # y = Array(2:n + 1)
    # y[3] = 100 
    y = [
        11.19
        10.52
        10.73
        10.26
        11.27
        11.54
        10.56
        10.48
        10.5
        10.34
        10.29
        10.43
        10.18
        10.43
        10.32
        10.14
        10.79
    ]

    x = [
        0.596
        0.604
        0.589
        0.589
        0.598
        0.595
        0.599
        0.601
        0.592
        0.600
        0.592
        0.587
        0.589
        0.588
        0.588
        0.586
        0.594
    ]

    # normalize
    x_min, x_max = minimum(x), maximum(x)
    x = (x .- x_min) ./ (x_max - x_min)
    y_min, y_max = minimum(y), maximum(y)
    y0 = (y .- y_min) ./ (y_max - y_min)
    n, m = size(x, 1), ndims(x)

    Plots.plotly()
    results = [lr_with_outlier.f(x, y0, n, m, i, x_min, x_max, y_min, y_max) for i in 1:2:n]
    plots = [i[1] for i in results]
    agg_p = Plots.plot(plots..., wsize = (1200, 900), show = false)
    # Plots.savefig(agg_p, "output_filename.html")
    # Plots.savefig(agg_p, "output_filename.png")
    Plots.savefig(agg_p, "figure.json")
end

if abspath(PROGRAM_FILE) == @__FILE__
    real_main()
end
end
