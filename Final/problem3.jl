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

cov = CSV.read("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/Final/problem3_cov.csv",DataFrame)
exp_rtn = CSV.read("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/Final/problem3_ER.csv",DataFrame)

er = [0.12913882911464453, 0.13155535179860567, 0.12392855449115958]
rf = 0.045
covar = [0.04568741078765268 0.018329837513673185 0.016714709708283582
        0.018329837513673185 0.04834944241137626 0.01719476719887282
        0.016714709708283582 0.01719476719887282 0.04020424552161158]
# covar = diagm(vols)*corel*diagm(vols)

function sr(w...)
    _w = collect(w)
    m = _w'*er - rf
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

n = 3

m = Model(Ipopt.Optimizer)
# set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
# @variable(m, w[i=1:n] ,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

wop = round.(value.(w),digits=4)


# Function for Portfolio Volatility
function pvol(w...)
    x = collect(w)
    return(sqrt(x'*covar*x))
end

# Function for Component Standard Deviation
function pCSD(w...)
    x = collect(w)
    pVol = pvol(w...)
    csd = x.*(covar*x)./pVol
    return (csd)
end

# Sum Square Error of cSD
function sseCSD(w...)
    csd = pCSD(w...)
    mCSD = sum(csd)/n
    dCsd = csd .- mCSD
    se = dCsd .*dCsd
    return(1.0e5*sum(se)) # Add a large multiplier for better convergence
end

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,sseCSD; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)
wrp = round.(value.(w),digits=4)


println("ER Optimal: $(wop'*er)")
println("SD Optimal: $(sqrt(wop'*covar*wop))")
println("SR Optimal: $((wop'*er - rf) / sqrt(wop'*covar*wop))")
println(" ")
println("ER RP: $(wrp'*er)")
println("SD RP: $(sqrt(wrp'*covar*wrp))")
println("SR Optimal: $((wrp'*er - rf) / sqrt(wrp'*covar*wrp))")


# correlation is the same
covar = [0.04568741078765268 0.018329837513673185 0.016714709708283582
        0.018329837513673185 0.04834944241137626 0.01719476719887282
        0.016714709708283582 0.01719476719887282 0.04020424552161158]
vols = [0.04568741078765268, 0.04834944241137626, 0.04020424552161158]
corr = cov2cor(covar, vols)