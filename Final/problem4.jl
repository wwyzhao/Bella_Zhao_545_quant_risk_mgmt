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

w = [0.2638965543426399, 0.21282173269180257, 0.5232817129655575]

stocks = [:Asset1, :Asset2, :Asset3]
upReturns = CSV.read("C:/Users/17337/Desktop/FinTech-545-Spring2023-main/Final/problem4_returns.csv",DataFrame)

#calculate portfolio return and updated weights for each day
n = size(upReturns,1)
m = size(stocks,1)

pReturn = Vector{Float64}(undef,n)
weights = Array{Float64,2}(undef,n,length(w))
lastW = copy(w)
matReturns = Matrix(upReturns[!,stocks])

for i in 1:n
    # Save Current Weights in Matrix
    weights[i,:] = lastW

    # Update Weights by return
    lastW = lastW .* (1.0 .+ matReturns[i,:])
    
    # Portfolio return is the sum of the updated weights
    pR = sum(lastW)
    # Normalize the wieghts back so sum = 1
    lastW = lastW / pR
    # Store the return
    pReturn[i] = pR - 1
end

# Set the portfolio return in the Update Return DataFrame
upReturns[!,:Portfolio] = pReturn

# Calculate the total return
totalRet = exp(sum(log.(pReturn .+ 1)))-1
# Calculate the Carino K
k = log(totalRet + 1 ) / totalRet

# Carino k_t is the ratio scaled by 1/K 
carinoK = log.(1.0 .+ pReturn) ./ pReturn / k
# Calculate the return attribution
attrib = DataFrame(matReturns .* weights .* carinoK, stocks)

# Set up a Dataframe for output.
Attribution = DataFrame(:Value => ["TotalReturn", "Return Attribution"])
# Loop over the stocks
for s in vcat(stocks,:Portfolio)
    # Total Stock return over the period
    tr = exp(sum(log.(upReturns[!,s] .+ 1)))-1
    # Attribution Return (total portfolio return if we are updating the portfolio column)
    atr =  s != :Portfolio ?  sum(attrib[:,s]) : tr
    # Set the values
    Attribution[!,s] = [ tr,  atr ]
end

# Realized Volatility Attribution

# Y is our stock returns scaled by their weight at each time
Y =  matReturns .* weights
# Set up X with the Portfolio Return
X = hcat(fill(1.0, size(pReturn,1)),pReturn)
# Calculate the Beta and discard the intercept
B = (inv(X'*X)*X'*Y)[2,:]
# Component SD is Beta times the standard Deviation of the portfolio
cSD = B * std(pReturn)

#Check that the sum of component SD is equal to the portfolio SD
sum(cSD) ≈ std(pReturn)
# Add the Vol attribution to the output 
Attribution = vcat(Attribution,    
    DataFrame(:Value=>"Vol Attribution", [Symbol(stocks[i])=>cSD[i] for i in 1:size(stocks,1)]... , :Portfolio=>std(pReturn))
)

println(Attribution)