@with_kw struct PCSS
    mᵈ::Int64
    Dmax::Int64 # maximum depth of solver
    PF::BasicParticleFilter   # particle filter
    δ::Float64
    ϵ::Union{Float64, Nothing}
    ϕ::String
end

@with_kw struct CCSS
    mᵈ::Int64
    Dmax::Int64 # maximum depth of solver
    PF::BasicParticleFilter   # particle filter
    δ::Float64
    ϕ::String
    scale::Bool = false
end


@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}  
    σʷ::Float64
    σᵛ::Float64 
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    Dmax::Int64
    λ::Float64
    rng::MersenneTwister
    a_space::Array{Float64, 2}
    beacons::Array{Float64, 2}
    obstacles::Array{Float64, 2}
    d::Float64
    rmin::Float64
    obsRadii::Float64
    goalRadii::Float64
    goal::Array{Float64, 1}
    rewardGoal::Float64
    rewardObs::Float64
    γ::Float64
    a_space_length::Int64
    obstacles_length::Int64
end



abstract type Planner end



struct PCSSPlanner <:Planner
    solver::PCSS 
    pomdp::POMDPscenario # problem
end


struct CCSSPlanner <:Planner
    solver::CCSS 
    pomdp::POMDPscenario # problem
    rng::MersenneTwister
end

struct NotFeasibleConstraintError <: Exception 
    var::String
end

Base.showerror(io::IO, e::NotFeasibleConstraintError) = print(io, e.var)


