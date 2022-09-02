using Base: Int64, debug_color
using Revise
using Distributions
using Random
using Plots
using LinearAlgebra
#using StatsPlots
using Parameters
#using TickTock
using ParticleFilters
using Statistics
using POMDPs
using MCTS
using QMDP
using ArgParse
using ProgressMeter
using Images
using POMDPTools
using Future


include("structs.jl") 
include("BeaconsWorld2D.jl")
#include("BeaconsWorld2DnoObstacles.jl")
include("utils.jl")
include("visualize.jl")


include("PCSS.jl")
include("CCSSimportance.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin  
        "--sol"
            help = "CCSS, PCSS"
            arg_type = String
            default = "PCSS"
        "--delta"
            help = "δ"
            arg_type = Float64
            default = 0.7
        "--epsilon"
            help = "ϵ"
            arg_type = Union{Float64, Nothing}
            default = 0.0
        "--obs"
            help = "observations to open"
            arg_type = Int64
            default = 5
        "--L"          
            help = "horizon"
            arg_type = Int64
            default = 3
        "--simulations"
            help = "number of simulations"
            arg_type = Int64
            default = 21
        "--Nstatistics"
            help = "number of repetitions"
            arg_type = Int64
            default = 50
        "--num-particles"
            help = "number of belief particles"
            arg_type = Int64
            default = 100
        "--constr-op"
            help = "Operator constraint"
            arg_type = String
            default = "Prob"    
    end
    return parse_args(s)
end



function main()
    parsed_args = parse_commandline()
    printstyled(" Parameters : $parsed_args \n \n"; color = :blue) 
    rng = MersenneTwister(81)

    cumulatives_no_collisions::Array{Float64,1}, cumulatives::Array{Float64,1}, iter_time::Array{Any,1}, collisions::Array{Any,1} = [], [], [], []
    Nstatisticscounter = 0
    #tick()
    @showprogress for i in 1:parsed_args["Nstatistics"]
        pomdp = BeaconsWorld2D(parsed_args["L"],rng)

        # create particle filter
        model = ParticleFilterModel{Vector{Float64}}((x...)-> dynamics(pomdp, x[1], x[2], x[3]), (x...)-> pdfObservationModel(pomdp, x[1], x[2], x[3], x[4]))
        N = parsed_args["num_particles"] # number of particles
        pf = BootstrapFilter(model, N, rng)
        
        # init trajectory (τ)
        x_gt_τ::Array{Vector{Float64},1}  = [initState()]
        b_gt_τ::Array{MvNormal,1}  = [initBelief()]
        b_PF_τ::Array{ParticleCollection{Vector{Float64}}} = [ParticleCollection([rand(pomdp.rng, b_gt_τ[1]) for i in 1:N])]
        r_gt_τ::Array{Float64,1}, o_gt_τ::Array{Vector{Float64},1}, policy_τ::Array{Vector{Float64},1} = Float64[], Vector{Float64}[], Vector{Float64}[]
        coll_τ = Bool[]
                    
        # define solver hyper params
        typeofB = typeof(b_PF_τ[end])
        typeofA = typeof(pomdp.a_space[1,:])
        

        if parsed_args["sol"] == "PCSS"
            solver = PCSS(;mᵈ=parsed_args["obs"], Dmax = pomdp.Dmax,PF = pf, 
                                   δ=parsed_args["delta"], ϵ=parsed_args["epsilon"], ϕ=parsed_args["constr_op"])
            planner = PCSSPlanner(solver, pomdp)    
            obstacles_title = "obstacles" 
        elseif parsed_args["sol"] == "CCSS"
            solver = CCSS(;mᵈ =parsed_args["obs"], Dmax = pomdp.Dmax,PF = pf, 
                                        δ=parsed_args["delta"], ϕ=parsed_args["constr_op"])
            planner = CCSSPlanner(solver, pomdp, Future.randjump(deepcopy(pomdp.rng), big(10)^20))    
            obstacles_title = "obstacles"     
        end   
   
        local a_best::typeofA
        local a_id_best::Int64
        prog =Progress(desc="simulations " *parsed_args["sol"], parsed_args["simulations"]-1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        
        try
            #tick()    
            for i in 1:parsed_args["simulations"]-1
                if parsed_args["constr_op"] == "CVaR"
                    status = _cvar_collision(planner, b_PF_τ[end])
                    println(status)
                end  
                if ((parsed_args["sol"] == "PCSS" && (planner.solver.ϵ != 1.0 &&  planner.solver.δ != 1.0)) ||
                   (parsed_args["sol"] == "CCSS" && (planner.solver.δ != 1.0)))  
                    status = _make_belief_safe!(planner.pomdp, b_PF_τ[end], pomdp.rng)
                    if status
                        printstyled("\n belief fixed \n ", color=:red)
                    end    
                end    
                  
                if  parsed_args["sol"] == "PCSS"
                    planner = PCSSPlanner(solver, pomdp)
                    (a_best, a_id_best) = planner(b_PF_τ[end])
                elseif  parsed_args["sol"] == "CCSS"
                    planner = CCSSPlanner(solver, pomdp, Future.randjump(deepcopy(pomdp.rng), big(10)^20))
                     (a_best, a_id_best) = planner(b_PF_τ[end], deepcopy(b_PF_τ[end]))           
                end    

                # perform action, get ground truth
                push!(policy_τ, a_best)
                (b_step, r_step, x_step, o_step, coll_status) = oneStepSim(planner, b_gt_τ[end], x_gt_τ[end], policy_τ[end])
   
                push!(b_gt_τ, b_step)
                push!(x_gt_τ, x_step)
                push!(r_gt_τ, r_step)
                push!(o_gt_τ , o_step)
                push!(coll_τ, coll_status)
 
                b_post = ParticleBeliefMDP(b_PF_τ[end], planner.pomdp, policy_τ[end], planner.solver.PF, o_gt_τ[end])
                push!(b_PF_τ, b_post)
                ProgressMeter.next!(prog; showvalues=[(:iter, i)])
            end
            ProgressMeter.finish!(prog)
            #runtime = peektimer()
            #append!(iter_time, runtime)
            #tock()

            cum = sum(r_gt_τ)
            if sum(coll_τ) > 0
                push!(collisions, 1)
                printstyled("Collision\n", color = :red)
            else
                push!(collisions, 0)
                append!(cumulatives_no_collisions, cum)    
            end    
            
            append!(cumulatives, cum)
            printstyled("∑rₜ is: $cum\n, iter: $i", color = :yellow)


            visualizeTrajectory(planner, x_gt_τ, b_gt_τ, b_PF_τ;obstacles_title=obstacles_title, alg_name=parsed_args["sol"], iter=i)
            Nstatisticscounter+=1
            #tock()    
        catch err
            println(err)
            @error "ERROR: " exception=(err, catch_backtrace())
            #rethrow(err)
            #tock()
            #tock()
        end    
    end
    #tock()
    if Nstatisticscounter == 0
        println("No sucessful repetitions")
    else
        #println("===============================")    
        #println("time per iteration: $iter_time")
        #println("===============================")
        collisions_number = sum(collisions)
        println("Number of falling tajectories is $collisions_number out of $(length(collisions))")
        mean_cum = mean(cumulatives)
        σ_cum = std(cumulatives)
        println("Mean cumulative reward (V): $mean_cum +- $σ_cum")
        mean_cum_no_coll = mean(cumulatives_no_collisions)
        σ_cum_no_coll = std(cumulatives_no_collisions)
        println("Mean cumulative reward no collisions (V): $mean_cum_no_coll +- $σ_cum_no_coll")
    end    
end




main()