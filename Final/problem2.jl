using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Roots
using Plots
using StatsPlots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization
using Printf
using StateSpaceModels


include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/RiskStats.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/simulate.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/fitted_model.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/return_calculate.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/gbsm.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/missing_cov.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/expost_factor.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/bt_american.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/ewCov.jl")
include("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/library/skewNormal.jl")

# Calculate the call price
# option_data = CSV.read("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/Final/problem2.csv",DataFrame)

rf = 0.045
b = rf-0.047039352232225926
S = 109.5363281813056
ttm = 151/255
strike = 98.5296691600875
ivol = 0.23
val = gbsm(true,S,strike,ttm,rf,b,ivol,includeGreeks=true)
println("Value: $(val.value)")
println("Delta: $(val.delta)")
println("Gamma: $(val.gamma)")
println("Vega: $(val.vega)")
println("Theta: $(val.theta)")
println("Rho: $(val.rho)")

vars = Vector{Float64}(undef,1000)
ess = Vector{Float64}(undef,1000)
S0 = S
for j in 1:1000
    d = Normal(0,ivol/sqrt(255))
    r = rand(d,1000)
    S = S0 .* (r .+ 1)
    p =  [gbsm(true,S[i],strike,150/255,rf,b,ivol).value for i in 1:1000]
    pnl = (S .- S0) - (p .- val.value)

    vars[j] = VaR(pnl)
    ess[j] = ES(pnl)
end

println("VaR Range: $(mean(vars)) 95% confidence [$(quantile(vars,.025)), $(quantile(vars,.975))]")
println("ES Range : $(mean(ess)) 95% confidence [$(quantile(ess,.025)), $(quantile(ess,.975))]")