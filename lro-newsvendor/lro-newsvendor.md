::: {.cell .markdown}
Newsvendor problem (1-d)
========================
:::

::: {.cell .markdown}
### Parameters

-   $h=b=1$
-   $F$ Underlying distribution
    -   $F_1 \sim N(50, 50)$
    -   $F_2 \sim \exp(1/50)$
-   $\mathbf N, |\mathbf N| = N$ is \# of realizations
:::

::: {.cell .markdown}
#### type of distribution set:

-   `likelihood`, see
    `Wang, Zizhuo, Peter W Glynn, and Yinyu Ye. 2016. “Likelihood Robust Optimization for Data-Driven Problems.” Computational Management Science 13 (2): 241–61. https://doi.org/10.1007/s10287-015-0240-3.`
:::

::: {.cell .markdown}
### The Models

#### Newsvendor primal

$$\begin{aligned} 
 \textsf{ minimize the worse-case expected cost: }  \\
 & \min_x \max_p \mathbb E_p(\mathbf h) \\
 \textsf{ loss function: }  \\
 &  \mathbf h, \quad h_i = b(d_i - x)^+ + h(x-d_i)^+\\
 \textbf {s.t. }  & \\
 & \mathbf p \in \mathcal D_d, \quad \textsf{ where } \mathcal D_d \textsf{ is some valid distribution set }
\end{aligned}$$

#### Scarf (DRO)

$$x_{\text {scar } f}^{*}=\hat{\mu}+\frac{\hat{\sigma}}{2}(\sqrt{\frac{b}{h}}-\sqrt{\frac{h}{b}})$$

we discuss the distribution set:
:::

::: {.cell .markdown}
#### LRO: log-likelihood

where

$$\begin{aligned} 
 \textsf{likehood: } & \\
 & \sum_{i=0}^{n} N_{i} \log p_{i} \geq \gamma \\ 
 & \sum_{i=0}^{n} p_{i}=1, \quad p_{i} \geq 0, \forall i
 \end{aligned}$$

we have the following:

$$\begin{aligned}
  & \max_{x, ...} 
  \theta + \beta \gamma + \beta N + t \\
  \textbf {s.t.} \\
  & (\mathbf q, \beta\mathbf N, t) \in \mathcal K_{\exp} \\
  & \beta \ge 0 \\
  & \mathbf q \equiv - \mathbf h - \theta \mathbf 1 \ge 0\\
  & \mathbf q + \theta \mathbf 1 +  b\cdot(d - x) \le 0\\
  & \mathbf q +  \theta \mathbf 1 + h\cdot(x - d) \le 0\\
  & x \in D 
\end{aligned}$$

using estimator:

$$\gamma^{*}=\sum_{i=1}^{n} N_{i} \log \frac{N_{i}}{N}-\frac{1}{2} \chi_{n-1,1-\alpha}^{2}$$

worst-case probability:

$$p^\star = \frac{\beta \mathbf N}{q}$$

#### exact moments

where $$\begin{aligned} 
  \textsf{moments (exact): } & \\
 & \sum_{i=0}^{n} d_{i} p_{i} = \mu \\ 
 & \sum_{i=0}^{n} d_{i}^2 p_{i} = \mu^2 +\sigma^2, \forall i, \quad \textsf{can use sample mean/var}
 \end{aligned}$$

$$\begin{aligned}
  & p^\star = \frac{\beta\mathbf N}{h - \theta \mathbf 1 - \alpha d - w(d\bullet d)}
\end{aligned}$$

$$\begin{aligned}
  & \max_{x, ...} 
  \theta + \beta \gamma + \alpha \mu + w(\hat \mu^2 + \hat \sigma^2) +\beta N + t \\
  \textbf {s.t.} \\
  & (\mathbf q, \beta\mathbf N, t) \in \mathcal K_{\exp} \\
  & \beta \ge 0 \\
  & \mathbf q \equiv - \mathbf h - \theta \mathbf 1 - ...\ge 0\\
  & \mathbf q + \theta \mathbf 1 + ...+  b\cdot(d - x) \le 0\\
  & \mathbf q +  \theta \mathbf 1 + ... + h\cdot(x - d) \le 0\\
  & x \in D 
\end{aligned}$$
:::

::: {.cell .markdown}
#### JuMP code

> Remark Julia MathOptInterface uses slight different notation on Cones,
> refer to [MathOptInterface
> API](https://jump.dev/MathOptInterface.jl/v0.9.1/apireference/#MathOptInterface.ExponentialCone)
:::

::: {.cell .code execution_count="1"}
``` {.julia}
using JuMP
using Distributions
using StatsBase
using MosekTools
using Plots
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

plotly()

# truncation @[0-200]
int_trunc = x -> round(min(max(x, 0), 200))

# sample object
struct Sample
    h::Float64
    b::Float64
    N::Int32
    n::Int32
    S
    H
    mu::Float64
    sig::Float64
end

# solution object
struct Sol
    x::Float64
    model::JuMP.Model
    p::Array{Float64,1}
end 

h = b = 1
N, n = 1000, 200
S1 = int_trunc.(rand(Normal(50, 50), N))
H1 = fit(Histogram, S1, 0:n)
mu1, sig1 = mean_and_std(S1)
S2 = int_trunc.(rand(Exponential(50), N))
H2 = fit(Histogram, S2, 0:n)
mu2, sig2 = mean_and_std(S2)
d = Array(1:200)

sample1 = Sample(h,b,N,n,S1, H1, mu1, sig1)
sample2 = Sample(h,b,N,n,S2, H2, mu2, sig2)

# simple lambda function
x_scarf = s -> s.mu
x_ro_1 = x_scarf(sample1)
x_ro_2 = x_scarf(sample2)
```

::: {.output .execute_result execution_count="1"}
    48.497
:::
:::

::: {.cell .code execution_count="2"}
``` {.julia}
# solution evaluation
mutable struct Eval
    sol::Sol
    sample::Sample
    d::Array
    
    # objectives
    obj_worse::Float64
    obj_true::Float64
    
    function Eval(sol::Sol, sample::Sample, d::Array)
        x = new(sol, sample, d, 0, 0)
        h = max.(
            (d .- sol.x) .* sample.b, 
            (sol.x .- d) .* sample.h
        )
        x.obj_true = sum(sample.H.weights .* h) / sample.N
        x.obj_worse = sum(sol.p .* h)
        x
    end
end
```
:::

::: {.cell .code execution_count="3" scrolled="true"}
``` {.julia}
# 1. pure lro model
function lro_nv_model(sample)
    h, b, N, n = sample.h, sample.b, sample.N, sample.n
    H = sample.H.weights
    Hs = [i for i in H if i > 0]
    gamma = sum(Hs .* (log.(Hs./N))) - 1/2 * quantile.(Gamma(n-1), [0.95])[1]
    model = JuMP.Model()
    @variable(model, theta)
    @variable(model, beta >= 0)
    @variable(model, q[1:n] >= 0)
    @variable(model, x >= 0)

    @constraint(model, q .+ b * (d .- x) .+ theta .<= 0)
    @constraint(model, q .+ h * (x .- d) .+ theta .<= 0)

    @variable(model, t[1:n])
    @constraint(model, KL_DEV[i=1:n], [t[i], H[i] * beta, q[i]] in MOI.ExponentialCone())
    obj_expr = 
    begin
        theta + beta * (gamma + N) + dot(ones(n), t)
    end
    @objective(model, Max, obj_expr)
    set_optimizer(model, Mosek.Optimizer)
    optimize!(model)
    x_sol = value.(x)
    p_sol = value.(beta).*H ./ value.(q)
    return Sol(x_sol, model, p_sol)
end
```

::: {.output .execute_result execution_count="3"}
    lro_nv_model (generic function with 1 method)
:::
:::

::: {.cell .code execution_count="4" scrolled="true"}
``` {.julia}
lro_sol1 = lro_nv_model(sample1)
# lro_sol2 = lro_nv_model(sample2)
# plot sampling distribution and worse-case


lro_p1 = plot(1:n, [sample1.H.weights lro_sol1.p * N],
    label=reshape(["@true", "@worst-case"], 1, 2),
    title="normal"
)
```

::: {.output .stream .stdout}
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1003            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Presolve started.
    Linear dependency checker started.
    Linear dependency checker terminated.
    Eliminator started.
    Freed constraints in eliminator : 0
    Eliminator terminated.
    Eliminator - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - number                 : 0               
    Presolve terminated. Time: 0.00    
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1003            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 12              
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 560
    Optimizer  - Cones                  : 201
    Optimizer  - Scalar variables       : 1003              conic                  : 602             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 1.49e+04          after factor           : 1.52e+04        
    Factor     - dense dim.             : 4                 flops                  : 1.58e+06        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   2.0e+02  2.2e+01  1.8e+02  0.00e+00   -1.835571490e+02  0.000000000e+00   1.0e+00  0.00  
    1   1.3e+02  1.4e+01  1.5e+02  -1.00e+00  -3.989615636e+02  -2.159883074e+02  6.6e-01  0.01  
    2   7.2e+01  7.7e+00  1.1e+02  -9.98e-01  -7.824056883e+02  -6.008989258e+02  3.6e-01  0.01  
    3   2.3e+01  2.5e+00  6.2e+01  -9.94e-01  -3.609534082e+03  -3.435781177e+03  1.2e-01  0.01  
    4   1.2e+01  1.3e+00  4.3e+01  -9.29e-01  -6.403360798e+03  -6.245779702e+03  6.1e-02  0.01  
    5   2.8e+00  2.9e-01  1.3e+01  -7.15e-01  -9.520306839e+03  -9.448357156e+03  1.4e-02  0.01  
    6   6.0e-01  6.5e-02  1.8e+00  3.07e-01   -2.608706123e+03  -2.589949885e+03  3.0e-03  0.01  
    7   9.2e-02  9.9e-03  1.1e-01  8.07e-01   -4.567125634e+02  -4.534855206e+02  4.6e-04  0.01  
    8   2.6e-02  2.7e-03  1.5e-02  1.00e+00   -1.768516712e+02  -1.759347962e+02  1.3e-04  0.01  
    9   8.1e-03  8.7e-04  2.5e-03  1.11e+00   -9.201516567e+01  -9.173400712e+01  4.0e-05  0.01  
    10  3.4e-03  3.6e-04  6.9e-04  9.65e-01   -6.915641721e+01  -6.903698977e+01  1.7e-05  0.01  
    11  1.6e-03  1.7e-04  2.1e-04  9.79e-01   -5.895075108e+01  -5.889592367e+01  7.7e-06  0.01  
    12  7.7e-04  8.2e-05  7.4e-05  1.01e+00   -5.475841261e+01  -5.473144813e+01  3.8e-06  0.02  
    13  3.3e-04  3.6e-05  2.1e-05  9.96e-01   -5.259406247e+01  -5.258235916e+01  1.7e-06  0.02  
    14  8.6e-05  9.2e-06  2.8e-06  9.92e-01   -5.134501011e+01  -5.134197255e+01  4.3e-07  0.02  
    15  8.2e-06  8.8e-07  8.2e-08  9.98e-01   -5.094834719e+01  -5.094805872e+01  4.1e-08  0.02  
    16  4.9e-07  5.2e-08  1.2e-09  9.99e-01   -5.090962356e+01  -5.090960643e+01  2.4e-09  0.02  
    17  1.1e-08  1.2e-09  4.2e-12  1.00e+00   -5.090725476e+01  -5.090725436e+01  5.6e-11  0.02  
    Optimizer terminated. Time: 0.04    
:::

::: {.output .display_data}
```{=html}
    <script type="text/javascript">
        requirejs(["https://cdn.plot.ly/plotly-latest.min.js"], function(p) {
            window.Plotly = p
        });
    </script>
```
:::

::: {.output .execute_result execution_count="4"}
``` {.json}
{"layout":{"annotations":[{"rotation":0,"showarrow":false,"text":"normal","xref":"paper","x":0.5222222222222223,"yanchor":"top","yref":"paper","font":{"color":"rgba(0, 0, 0, 1.000)","size":20,"family":"sans-serif"},"xanchor":"center","y":1}],"height":400,"yaxis":{"zeroline":false,"showline":true,"ticktext":["0","50","100","150"],"domain":[3.762029746281716e-2,0.9415463692038496],"showgrid":true,"titlefont":{"color":"rgba(0, 0, 0, 1.000)","size":15,"family":"sans-serif"},"showticklabels":true,"visible":true,"tickfont":{"color":"rgba(0, 0, 0, 1.000)","size":11,"family":"sans-serif"},"zerolinecolor":"rgba(0, 0, 0, 1.000)","anchor":"x1","tickangle":0,"range":[-5.38671427660452,184.94385683008855],"gridcolor":"rgba(0, 0, 0, 0.100)","tickvals":[0,50,100,150],"title":"","tickcolor":"rgb(0, 0, 0)","type":"-","linecolor":"rgba(0, 0, 0, 1.000)","ticks":"inside","tickmode":"array","gridwidth":0.5,"mirror":false},"showlegend":true,"xaxis":{"zeroline":false,"showline":true,"ticktext":["0","50","100","150","200"],"domain":[5.100612423447069e-2,0.9934383202099737],"showgrid":true,"titlefont":{"color":"rgba(0, 0, 0, 1.000)","size":15,"family":"sans-serif"},"showticklabels":true,"visible":true,"tickfont":{"color":"rgba(0, 0, 0, 1.000)","size":11,"family":"sans-serif"},"zerolinecolor":"rgba(0, 0, 0, 1.000)","anchor":"y1","tickangle":0,"range":[-4.97,205.97],"gridcolor":"rgba(0, 0, 0, 0.100)","tickvals":[0,50,100,150,200],"title":"","tickcolor":"rgb(0, 0, 0)","type":"-","linecolor":"rgba(0, 0, 0, 1.000)","ticks":"inside","tickmode":"array","gridwidth":0.5,"mirror":false},"width":600,"margin":{"l":0,"t":20,"b":20,"r":0},"paper_bgcolor":"rgba(255, 255, 255, 1.000)","plot_bgcolor":"rgba(255, 255, 255, 1.000)","legend":{"bgcolor":"rgba(255, 255, 255, 1.000)","x":1,"tracegroupgap":0,"font":{"color":"rgba(0, 0, 0, 1.000)","size":11,"family":"sans-serif"},"bordercolor":"rgba(0, 0, 0, 1.000)","y":1}},"data":[{"colorbar":{"title":""},"yaxis":"y1","showlegend":true,"mode":"lines","xaxis":"x1","line":{"color":"rgba(0, 154, 250, 1.000)","width":1,"shape":"linear","dash":"solid"},"name":"@true","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200],"zmin":null,"type":"scatter","zmax":null,"legendgroup":"@true","y":[163,7,7,5,6,5,7,6,5,8,7,5,7,9,11,10,5,9,5,8,6,3,7,8,4,3,6,6,7,5,3,4,8,7,9,5,8,7,10,7,9,3,6,6,7,9,12,4,8,6,10,8,8,2,6,14,6,7,3,7,10,11,4,1,11,6,6,11,8,10,7,11,7,5,6,4,8,13,6,4,6,11,5,11,8,4,5,6,7,5,5,9,4,1,3,8,4,8,7,7,2,7,5,5,5,1,10,6,3,3,6,2,3,4,6,3,3,3,1,3,3,5,4,3,2,1,4,1,1,1,1,2,4,4,4,1,0,4,0,1,3,4,1,2,0,0,0,2,2,1,1,2,1,0,0,3,1,1,0,0,2,2,0,1,2,0,2,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0]},{"colorbar":{"title":""},"yaxis":"y1","showlegend":true,"mode":"lines","xaxis":"x1","line":{"color":"rgba(227, 111, 71, 1.000)","width":1,"shape":"linear","dash":"solid"},"name":"@worst-case","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200],"zmin":null,"type":"scatter","zmax":null,"legendgroup":"@worst-case","y":[179.55714255348403,7.617256615599037,7.525722622553834,5.31168790526836,6.299228255323154,5.188472316530552,7.18057639984486,6.085011997998033,5.014006923259489,7.933487381343173,6.865700258436011,4.850892835965498,6.718395752727454,8.546256424361605,10.335723833942598,9.298458083083922,4.6014069762813765,8.198203129294868,4.508653309479396,7.141862724068933,5.30347788585389,2.625797607787855,6.067500618604502,6.867749726481849,3.401239698228746,2.526914076850712,5.006690693602237,4.960425743782115,5.73417577036943,4.058678897855441,2.4133120189114927,3.18907479083044,6.321813023687475,5.48315639419722,6.988585642917328,3.8491406469801577,6.106084234020931,5.297629112676372,7.504560233744957,5.209495252332997,6.642666833464777,2.1961060772747567,4.356565018801954,4.321493031513662,5.00147788234246,6.379523787945341,8.439169301562073,2.7911175798753143,5.539033759345165,4.122373105124852,6.8182605870374005,5.413353995585297,5.372718628464775,1.333173023263332,3.9699385423799494,9.195185514202164,3.912075205838232,4.5310665303541064,1.927936930139333,4.4733875840358,6.436525965947942,7.131483915480636,2.6121965608814928,0.6578518249228695,7.289961035314904,4.006017120294695,4.036137531189118,7.455641551258222,5.463676724370931,6.882130821616819,4.854836907077882,7.688630547380946,4.931291009382499,3.550306252760962,4.29445017212763,2.886055649166049,5.81903773101223,9.533442709673649,4.43641543575403,2.982257473625129,4.510977018634206,8.340208998455415,3.823406657593347,8.484005712842535,6.223840209606514,3.139218356430616,3.958748456002338,4.792913012966574,5.642107553854771,4.066714301530057,4.104023601011587,7.455641669165559,3.3445877426463047,0.8440361354971719,2.5562209872980572,6.8821309074465,3.4744746237804773,7.017074633838866,6.200731983638254,6.262739294441876,1.8074294592597804,6.39055028224039,4.61173777982975,4.659776707738745,4.708826982578159,0.9517854009537752,9.620182371315845,5.8348504557250385,2.949485687439731,2.9822577457727437,6.031530778555215,2.033358098509342,3.0850942035862996,4.161289267970764,6.315367478288135,3.1952761220121983,3.233773418279687,3.27320967310465,1.1045410479824804,3.355039901637411,3.3975087542366267,5.735109620775166,4.647673652080794,3.5316209210013483,2.3858067722585674,1.2090246398651403,4.902340660056316,1.242608651412787,1.2601101785310398,1.2781117492845049,1.2966351046010771,2.6314044842028794,5.341356163914958,5.422285788164206,5.5057055546551705,1.3979347034468044,0,5.772110610711956,0,1.4911303326074028,4.549206415167074,6.17018712197775,1.5696108603649468,3.1952767459743763,0,0,0,3.441067213039825,3.508539110757653,1.789356323579214,1.8258737889949115,3.7278227673019266,1.9035705218716905,0,0,6.100071845018958,2.0806467895025977,2.130185983382043,0,0,4.588089321799631,4.708828484473441,0,2.4852169125297707,5.112442250270646,0,5.422287153110956,0,0,0,3.0850966390603904,0,0,0,0,0,0,4.066717908538319,0,4.473389527719681,0,0,0,0,0,0,6.8821360327960415,0,0,0,0,11.183466191190988,0,0,0,0,0,0,0,0]}]}
```
:::
:::

::: {.cell .code execution_count="5"}
``` {.julia}
# 2. lro + moments model
function lro_moment_nv_model(sample::Sample)
    h, b, N, n = sample.h, sample.b, sample.N, sample.n
    H, u, sig = sample.H.weights, sample.mu, sample.sig 
    Hs = [i for i in H if i > 0]
    gamma = sum(Hs .* (log.(Hs./N))) - 1/2 * quantile.(Gamma(n-1), [0.95])[1]
    model = JuMP.Model()
    @variable(model, theta)
    @variable(model, beta >= 0)
    @variable(model, q[1:n] >= 0)
    @variable(model, x >= 0)
    @variable(model, a)
    @variable(model, w)

    @constraint(model, q .+ b * (d .- x) .+ theta .+ (d .* a) .+ (d .* d .* w) .<= 0)
    @constraint(model, q .+ h * (x .- d) .+ theta .+ (d .* a) .+ (d .* d .* w) .<= 0)

    @variable(model, t[1:n])
    @constraint(model, KL_DEV[i=1:n], [t[i], H[i] * beta, q[i]] in MOI.ExponentialCone())
    obj_expr = 
    begin
        theta + a * u + w * (u^2 + sig^2) + beta * (gamma + N) + dot(ones(n), t)
    end
    @objective(model, Max, obj_expr)
    set_optimizer(model, Mosek.Optimizer)
    optimize!(model)
    x_sol = value.(x)
    p_sol = value.(beta) .* H ./ value.(q)
    return Sol(x_sol, model, p_sol)
end
```

::: {.output .execute_result execution_count="5"}
    lro_moment_nv_model (generic function with 1 method)
:::
:::

::: {.cell .markdown}
#### Wrap up results
:::

::: {.cell .code execution_count="6"}
``` {.julia}
samples = Dict(
    "normal" => sample1, 
    "exp" => sample2
)
models = Dict(
    "lro" => lro_nv_model,
    "lro_mm" => lro_moment_nv_model
)
```

::: {.output .execute_result execution_count="6"}
    Dict{String,Function} with 2 entries:
      "lro_mm" => lro_moment_nv_model
      "lro"    => lro_nv_model
:::
:::

::: {.cell .code execution_count="7" scrolled="true"}
``` {.julia}
data = []
results = Dict()
data = [(v = samples[k]; 
        eval = Eval(models[m](v), v, d); 
        results[k, m] = eval;
        a = [eval.sol.x eval.obj_true eval.obj_worse]) 
    for k in ["normal", "exp"] for m in ["lro", "lro_mm"]]
data = vcat(data...)
```

::: {.output .stream .stdout}
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1003            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Presolve started.
    Linear dependency checker started.
    Linear dependency checker terminated.
    Eliminator started.
    Freed constraints in eliminator : 0
    Eliminator terminated.
    Eliminator - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - number                 : 0               
    Presolve terminated. Time: 0.00    
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1003            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 12              
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 560
    Optimizer  - Cones                  : 201
    Optimizer  - Scalar variables       : 1003              conic                  : 602             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 1.49e+04          after factor           : 1.52e+04        
    Factor     - dense dim.             : 4                 flops                  : 1.58e+06        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   2.0e+02  2.2e+01  1.8e+02  0.00e+00   -1.835571490e+02  0.000000000e+00   1.0e+00  0.00  
    1   1.3e+02  1.4e+01  1.5e+02  -1.00e+00  -3.989615636e+02  -2.159883074e+02  6.6e-01  0.00  
    2   7.2e+01  7.7e+00  1.1e+02  -9.98e-01  -7.824056883e+02  -6.008989258e+02  3.6e-01  0.01  
    3   2.3e+01  2.5e+00  6.2e+01  -9.94e-01  -3.609534082e+03  -3.435781177e+03  1.2e-01  0.01  
    4   1.2e+01  1.3e+00  4.3e+01  -9.29e-01  -6.403360798e+03  -6.245779702e+03  6.1e-02  0.01  
    5   2.8e+00  2.9e-01  1.3e+01  -7.15e-01  -9.520306839e+03  -9.448357156e+03  1.4e-02  0.01  
    6   6.0e-01  6.5e-02  1.8e+00  3.07e-01   -2.608706123e+03  -2.589949885e+03  3.0e-03  0.01  
    7   9.2e-02  9.9e-03  1.1e-01  8.07e-01   -4.567125634e+02  -4.534855206e+02  4.6e-04  0.01  
    8   2.6e-02  2.7e-03  1.5e-02  1.00e+00   -1.768516712e+02  -1.759347962e+02  1.3e-04  0.01  
    9   8.1e-03  8.7e-04  2.5e-03  1.11e+00   -9.201516567e+01  -9.173400712e+01  4.0e-05  0.01  
    10  3.4e-03  3.6e-04  6.9e-04  9.65e-01   -6.915641721e+01  -6.903698977e+01  1.7e-05  0.01  
    11  1.6e-03  1.7e-04  2.1e-04  9.79e-01   -5.895075108e+01  -5.889592367e+01  7.7e-06  0.01  
    12  7.7e-04  8.2e-05  7.4e-05  1.01e+00   -5.475841261e+01  -5.473144813e+01  3.8e-06  0.01  
    13  3.3e-04  3.6e-05  2.1e-05  9.96e-01   -5.259406247e+01  -5.258235916e+01  1.7e-06  0.01  
    14  8.6e-05  9.2e-06  2.8e-06  9.92e-01   -5.134501011e+01  -5.134197255e+01  4.3e-07  0.01  
    15  8.2e-06  8.8e-07  8.2e-08  9.98e-01   -5.094834719e+01  -5.094805872e+01  4.1e-08  0.01  
    16  4.9e-07  5.2e-08  1.2e-09  9.99e-01   -5.090962356e+01  -5.090960643e+01  2.4e-09  0.01  
    17  1.1e-08  1.2e-09  4.2e-12  1.00e+00   -5.090725476e+01  -5.090725436e+01  5.6e-11  0.02  
    Optimizer terminated. Time: 0.02    

    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1005            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Presolve started.
    Linear dependency checker started.
    Linear dependency checker terminated.
    Eliminator started.
    Freed constraints in eliminator : 0
    Eliminator terminated.
    Eliminator - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - number                 : 0               
    Presolve terminated. Time: 0.00    
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1005            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 12              
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 560
    Optimizer  - Cones                  : 201
    Optimizer  - Scalar variables       : 1005              conic                  : 604             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 1.57e+04          after factor           : 1.64e+04        
    Factor     - dense dim.             : 6                 flops                  : 1.64e+06        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   2.3e+00  4.9e+03  1.8e+02  0.00e+00   -1.835571490e+02  0.000000000e+00   1.0e+00  0.00  
    1   1.4e+00  3.0e+03  1.4e+02  -8.85e-01  -1.759083628e+02  1.253673382e-01   6.0e-01  0.00  
    2   5.2e-01  1.1e+03  7.3e+01  -8.12e-01  -1.493340599e+02  8.721907858e-02   2.3e-01  0.01  
    3   3.2e-01  6.9e+02  4.7e+01  -4.62e-01  -1.241687344e+02  4.918787713e-02   1.4e-01  0.01  
    4   9.8e-02  2.1e+02  1.3e+01  -1.80e-01  -6.145318972e+01  -4.546293609e-01  4.3e-02  0.01  
    5   4.9e-02  1.1e+02  5.0e+00  4.30e-01   -3.717216780e+01  -1.689709475e+00  2.1e-02  0.01  
    6   3.6e-02  7.9e+01  3.3e+00  6.52e-01   -3.036190665e+01  -2.821187711e+00  1.6e-02  0.01  
    7   1.3e-02  2.7e+01  7.3e-01  7.18e-01   -1.723287775e+01  -6.800197966e+00  5.5e-03  0.01  
    8   4.9e-03  1.1e+01  1.9e-01  8.55e-01   -1.694069638e+01  -1.262668888e+01  2.2e-03  0.01  
    9   1.6e-03  3.5e+00  3.8e-02  9.17e-01   -2.487301088e+01  -2.343371479e+01  7.0e-04  0.01  
    10  5.9e-04  1.3e+00  8.7e-03  9.65e-01   -3.437238333e+01  -3.383444163e+01  2.6e-04  0.01  
    11  3.3e-04  7.1e-01  3.7e-03  9.13e-01   -3.679711609e+01  -3.648876839e+01  1.4e-04  0.01  
    12  1.1e-04  2.4e-01  7.7e-04  9.28e-01   -3.906203099e+01  -3.895386696e+01  4.9e-05  0.01  
    13  4.2e-05  9.0e-02  1.8e-04  9.42e-01   -3.992301273e+01  -3.988173061e+01  1.8e-05  0.01  
    14  1.7e-05  3.6e-02  4.8e-05  9.38e-01   -4.021501176e+01  -4.019783292e+01  7.4e-06  0.01  
    15  8.5e-06  1.8e-02  1.8e-05  9.62e-01   -4.031308332e+01  -4.030427480e+01  3.7e-06  0.01  
    16  5.2e-06  1.1e-02  8.5e-06  9.86e-01   -4.035941179e+01  -4.035402228e+01  2.3e-06  0.02  
    17  4.4e-07  9.4e-04  2.1e-07  9.95e-01   -4.043109274e+01  -4.043064209e+01  1.9e-07  0.02  
    18  2.1e-08  5.4e-05  2.9e-09  9.95e-01   -4.043903227e+01  -4.043900654e+01  1.1e-08  0.02  
    19  7.1e-10  1.8e-06  1.8e-11  9.96e-01   -4.043952255e+01  -4.043952169e+01  3.6e-10  0.02  
    20  5.6e-11  1.8e-08  1.7e-14  1.00e+00   -4.043954026e+01  -4.043954025e+01  3.6e-12  0.02  
    Optimizer terminated. Time: 0.02    

    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1003            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Presolve started.
    Linear dependency checker started.
    Linear dependency checker terminated.
    Eliminator started.
    Freed constraints in eliminator : 0
    Eliminator terminated.
    Eliminator - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - number                 : 0               
    Presolve terminated. Time: 0.00    
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1003            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 12              
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 557
    Optimizer  - Cones                  : 201
    Optimizer  - Scalar variables       : 1003              conic                  : 602             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 1.44e+04          after factor           : 1.47e+04        
    Factor     - dense dim.             : 4                 flops                  : 1.49e+06        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   2.0e+02  2.7e+02  3.8e+02  0.00e+00   -3.813651763e+02  0.000000000e+00   1.0e+00  0.00  
    1   6.1e+01  8.1e+01  2.1e+02  -9.99e-01  -1.575752734e+03  -1.197478623e+03  3.0e-01  0.00  
    2   1.0e+01  1.3e+01  8.2e+01  -9.87e-01  -1.049503045e+04  -1.015257754e+04  4.9e-02  0.01  
    3   2.7e+00  3.6e+00  2.4e+01  -6.06e-01  -6.269265671e+03  -6.102343978e+03  1.3e-02  0.01  
    4   8.1e-01  1.1e+00  4.9e+00  2.50e-01   -2.625761751e+03  -2.561054962e+03  4.0e-03  0.01  
    5   1.5e-01  2.0e-01  4.2e-01  7.12e-01   -6.382790202e+02  -6.247756165e+02  7.5e-04  0.01  
    6   3.2e-02  4.3e-02  4.0e-02  9.79e-01   -1.945166950e+02  -1.916269352e+02  1.6e-04  0.01  
    7   1.0e-02  1.3e-02  7.2e-03  1.06e+00   -1.048601225e+02  -1.039728871e+02  5.0e-05  0.01  
    8   4.0e-03  5.3e-03  2.0e-03  9.06e-01   -7.478417327e+01  -7.440746021e+01  2.0e-05  0.01  
    9   1.9e-03  2.5e-03  6.8e-04  8.90e-01   -6.182279303e+01  -6.163627527e+01  9.5e-06  0.01  
    10  1.0e-03  1.3e-03  2.6e-04  9.74e-01   -5.611241508e+01  -5.601460415e+01  5.0e-06  0.01  
    11  4.8e-04  6.4e-04  8.6e-05  9.90e-01   -5.291059352e+01  -5.286350318e+01  2.4e-06  0.01  
    12  1.2e-04  1.7e-04  1.1e-05  9.95e-01   -5.071712829e+01  -5.070490733e+01  6.2e-07  0.01  
    13  3.5e-05  4.6e-05  1.7e-06  9.98e-01   -5.016046431e+01  -5.015705207e+01  1.7e-07  0.01  
    14  2.4e-06  3.2e-06  3.0e-08  1.00e+00   -4.995942500e+01  -4.995919178e+01  1.2e-08  0.01  
    15  1.8e-07  1.1e-07  1.9e-10  1.00e+00   -4.994518262e+01  -4.994517471e+01  4.0e-10  0.01  
    16  9.7e-08  2.7e-09  7.5e-13  1.00e+00   -4.994467663e+01  -4.994467643e+01  1.0e-11  0.02  
    17  6.4e-09  1.8e-10  1.3e-14  1.00e+00   -4.994466488e+01  -4.994466487e+01  6.7e-13  0.02  
    Optimizer terminated. Time: 0.02    

    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1005            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Presolve started.
    Linear dependency checker started.
    Linear dependency checker terminated.
    Eliminator started.
    Freed constraints in eliminator : 0
    Eliminator terminated.
    Eliminator - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - number                 : 0               
    Presolve terminated. Time: 0.00    
    Problem
      Name                   :                 
      Objective sense        : max             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 1000            
      Cones                  : 200             
      Scalar variables       : 1005            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 12              
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 557
    Optimizer  - Cones                  : 201
    Optimizer  - Scalar variables       : 1005              conic                  : 604             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 1.52e+04          after factor           : 1.59e+04        
    Factor     - dense dim.             : 6                 flops                  : 1.56e+06        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   2.3e+00  4.4e+03  3.8e+02  0.00e+00   -3.813651763e+02  0.000000000e+00   1.0e+00  0.00  
    1   1.3e+00  2.5e+03  2.7e+02  -8.67e-01  -3.612671151e+02  9.145486696e-02   5.6e-01  0.00  
    2   5.7e-01  1.1e+03  1.6e+02  -7.72e-01  -3.117447162e+02  9.139371151e-02   2.5e-01  0.01  
    3   3.5e-01  6.8e+02  1.0e+02  -4.54e-01  -2.612883875e+02  5.589665016e-02   1.5e-01  0.01  
    4   6.0e-02  1.2e+02  1.5e+01  -1.85e-01  -8.345479093e+01  -5.192001803e-01  2.6e-02  0.01  
    5   8.1e-03  1.6e+01  8.9e-01  6.33e-01   -1.739752790e+01  -4.162087409e+00  3.5e-03  0.01  
    6   3.5e-03  6.7e+00  2.5e-01  9.53e-01   -2.149219410e+01  -1.576260441e+01  1.5e-03  0.01  
    7   1.1e-03  2.2e+00  4.7e-02  9.50e-01   -3.092014255e+01  -2.897653521e+01  5.0e-04  0.01  
    8   5.7e-04  1.1e+00  1.8e-02  8.81e-01   -3.431858027e+01  -3.328082373e+01  2.5e-04  0.01  
    9   2.5e-04  4.8e-01  5.5e-03  8.11e-01   -3.619043680e+01  -3.569264375e+01  1.1e-04  0.01  
    10  1.2e-04  2.3e-01  1.9e-03  8.24e-01   -3.701229705e+01  -3.675630766e+01  5.2e-05  0.01  
    11  5.8e-05  1.1e-01  6.5e-04  8.95e-01   -3.750361361e+01  -3.737613872e+01  2.5e-05  0.01  
    12  2.6e-05  5.0e-02  2.0e-04  9.41e-01   -3.779593643e+01  -3.773809059e+01  1.1e-05  0.02  
    13  8.9e-06  1.7e-02  4.0e-05  9.84e-01   -3.798025224e+01  -3.796032417e+01  3.9e-06  0.02  
    14  4.7e-07  9.2e-04  5.0e-07  9.97e-01   -3.807826649e+01  -3.807719987e+01  2.1e-07  0.02  
    15  1.4e-08  4.0e-05  4.5e-09  9.96e-01   -3.808482943e+01  -3.808478334e+01  8.9e-09  0.02  
    16  5.0e-09  9.3e-07  1.6e-11  1.00e+00   -3.808514392e+01  -3.808514285e+01  2.1e-10  0.02  
    17  1.1e-09  8.0e-08  4.1e-13  1.00e+00   -3.808515040e+01  -3.808515030e+01  1.8e-11  0.02  
    Optimizer terminated. Time: 0.02    
:::

::: {.output .execute_result execution_count="7"}
    4×3 Array{Float64,2}:
     59.8911  36.9653  42.6789
     53.9407  36.5232  40.4394
     47.0     31.793   49.9453
     32.0     30.645   38.0847
:::
:::

::: {.cell .code execution_count="8"}
``` {.julia}
objective_value(results["exp", "lro_mm"].sol.model)
```

::: {.output .execute_result execution_count="8"}
    -38.08515039579801
:::
:::

::: {.cell .code execution_count="11"}
``` {.julia}
using DataFrames
```

::: {.output .stream .stderr}
    ┌ Info: Precompiling DataFrames [a93c6f00-e57d-5684-b7b6-d8193f3e46c0]
    └ @ Base loading.jl:1260
:::
:::

::: {.cell .code execution_count="12"}
``` {.julia}
data
```

::: {.output .execute_result execution_count="12"}
    4×3 Array{Float64,2}:
     59.8911  36.9653  42.6789
     53.9407  36.5232  40.4394
     47.0     31.793   49.9453
     32.0     30.645   38.0847
:::
:::

::: {.cell .code execution_count="17"}
``` {.julia}
df = DataFrame(
    Index=["normal, LRO", "normal, LRO_mm", "Exp, LRO", "Exp, LRO_mm"], 
    Sol=data[:, 1],
    True_obj=data[:, 2],
    Worst_obj=data[:, 3],
)
```

::: {.output .execute_result execution_count="17"}
```{=html}
<table class="data-frame"><thead><tr><th></th><th>Index</th><th>Sol</th><th>True_obj</th><th>Worst_obj</th></tr><tr><th></th><th>String</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>4 rows × 4 columns</p><tr><th>1</th><td>normal, LRO</td><td>59.8911</td><td>36.9653</td><td>42.6789</td></tr><tr><th>2</th><td>normal, LRO_mm</td><td>53.9407</td><td>36.5232</td><td>40.4394</td></tr><tr><th>3</th><td>Exp, LRO</td><td>47.0</td><td>31.793</td><td>49.9453</td></tr><tr><th>4</th><td>Exp, LRO_mm</td><td>32.0</td><td>30.645</td><td>38.0847</td></tr></tbody></table>
```
:::
:::
