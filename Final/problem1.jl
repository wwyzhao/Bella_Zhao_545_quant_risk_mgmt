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

# calculate log returns
prices = CSV.read("problem1.csv",DataFrame)
returns = return_calculate(prices,method="LOG",dateColumn="Date")

# Calculate Pairwise Covariance
col = [:Price1, :Price2, :Price3]
x = Matrix(prices[!,col])
pairwise = missing_cov(x,skipMiss=false,fun=cov)

# is the matrix PSD
eVal = eigvals(pairwise)

