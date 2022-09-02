function PCSS(π::PCSSPlanner, b_PF::ParticleCollection{Vector{Float64}})::Tuple{Vector{Float64}, Int64} 

 
        local a::Union{Vector{Float64},Nothing}
        local a_idx::Union{Int64, Nothing}
      
        best = _sparse_sampling(π, b_PF, π.solver.Dmax)
        a = best.a 
        a_idx = best.a_idx

    

    return a, a_idx
end


# ϵ = 0
function _sparse_sampling(planner::PCSSPlanner, 
                          b::ParticleCollection{Vector{Float64}}, 
                          d::Int64)
    φ = _prob_safe(planner, b)
    indicator = φ ≥ planner.solver.δ
    r = _mean_distance_to_goal(planner,b)
    if d ≤ 0
        return (a=nothing, a_idx=nothing, u = r, c=[indicator])
    end   
    best = (a=nothing, a_idx=nothing, u=-Inf, c=[false])

      
    

    for i in 1:planner.pomdp.a_space_length
        #if i == planner.pomdp.a_space_length
        #    printstyled("\nDebug: Last action reached at the level $d \n", color=:red)
        #end
        a = planner.pomdp.a_space[i,:]
        u = 0.0
    
        bpropagated = ParticleCollection(predict(planner.solver.PF, b, a, planner.pomdp.rng))
        
        this_level_indicators = Bool[]
        
        for _ in 1:planner.solver.mᵈ
            xgt = rand(planner.pomdp.rng, particles(bpropagated))
            obs = SampleObservation(planner, xgt, planner.pomdp.rng)
            bposterior= posterior(planner, b, a, bpropagated, obs,planner.pomdp.rng)
         
            best′ = _sparse_sampling(planner, bposterior, d-1)
            u′ , c′  = best′.u, best′.c
            #if isnothing(c′)
            #    printstyled("\nDebug: The variable c′ is nothing \n", color=:red)
            #end
            u += (r + planner.pomdp.γ*u′) / planner.solver.mᵈ
            
            append!(this_level_indicators, indicator .&& c′)   
        end
        status = sum(this_level_indicators) == length(this_level_indicators)   
        if status && u > best.u     
            best = (a=a, a_idx=i, u=u, c=this_level_indicators)            
        end    
    end
    if isnothing(best.a) && d == planner.solver.Dmax
        throw(NotFeasibleConstraintError("The constraint is not feasible at depth $d")) 
    end    

    return best
end

(π::PCSSPlanner)(b::ParticleCollection{Vector{Float64}}) = PCSS(π, b)