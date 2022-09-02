function CCSS(π::CCSSPlanner,b_PF::ParticleCollection{Vector{Float64}},
              b_constr::ParticleCollection{Vector{Float64}})::Tuple{Vector{Float64}, Int64} 

        
    local a::Union{Vector{Float64},Nothing}
    local a_idx::Union{Int64, Nothing}
      
    best = _sparse_sampling(π, b_PF, b_constr, π.solver.Dmax)
    a = best.a 
    a_idx = best.a_idx
        
    return a, a_idx
end



function _sparse_sampling(planner::CCSSPlanner, 
                          b::ParticleCollection{Vector{Float64}},
                          b̄::ParticleCollection{Vector{Float64}}, 
                          d::Int64)
    φ = _prob_safe(planner, b̄) 
    r = _mean_distance_to_goal(planner,b)
    if d ≤ 0
        return (a=nothing, a_idx=nothing, v = r, φ=φ)
    end   
    best = (a=nothing, a_idx=nothing, v=-Inf, φ=0.0)
    _make_belief_safe!(planner.pomdp, b̄, planner.rng)
      
    
    for i in 1:planner.pomdp.a_space_length
        a = planner.pomdp.a_space[i,:]
        v = 0.0
        obs_weights = []
        b_prop_prime = ParticleCollection(predict(planner.solver.PF, b, a, planner.pomdp.rng))    
        b̄_prop_prime = ParticleCollection(predict(planner.solver.PF, b̄, a, planner.rng))
        
        this_level_expected_φ = Float64[]
        # loop over the observations 
        # ================================================================================== #
        for _ in 1:planner.solver.mᵈ
            xᵒ = rand(planner.pomdp.rng, particles(b_prop_prime))
            obs = SampleObservation(planner, xᵒ, planner.pomdp.rng)

            wᶻ=__calculate_obs_weight(planner.pomdp, b_prop_prime, b̄_prop_prime, obs)
            push!(obs_weights, wᶻ)
            bposterior = posterior(planner, b, a, b_prop_prime, obs, planner.pomdp.rng)
            b̄posterior = posterior(planner, b̄, a, b̄_prop_prime, obs, planner.rng)


            
            best′ = _sparse_sampling(planner, bposterior, b̄posterior , d-1)
            
            
            v′ , φ′  = best′.v, best′.φ

            v += (r + planner.pomdp.γ*v′) / planner.solver.mᵈ
            


            append!(this_level_expected_φ, φ′)   
        end
        # ================================================================================== #
        expected_φ = dot(obs_weights, this_level_expected_φ)/sum(obs_weights)
        if planner.solver.scale 
            status =  expected_φ ≥ (planner.solver.δ)^d
        else 
            status = expected_φ  ≥ planner.solver.δ
        end   
         
        if status  && v > best.v     
            best = (a=a, a_idx=i, v=v, φ = φ ⋅ expected_φ)            
        end    
    end
    if isnothing(best.a) && d == planner.solver.Dmax
        throw(NotFeasibleConstraintError("The constraint is not feasible at depth $d")) 
    end    

    return best
end

(π::CCSSPlanner)(b::ParticleCollection{Vector{Float64}}, bsafe::ParticleCollection{Vector{Float64}}) = CCSS(π, b,bsafe)