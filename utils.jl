# propagate belief
function ParticleBeliefMDP(b::ParticleCollection{Vector{Float64}}, pomdp::POMDPscenario, a::Array{Float64, 1}, pf, o::Array{Float64, 1})::ParticleCollection{Vector{Float64}}
    bp = predict(pf, b, a, pomdp.rng)
    b_post = update(pf, b, a, o)
    return b_post
end

function posterior(p::Planner, 
                   b::ParticleCollection{Vector{Float64}}, 
                   a::Array{Float64, 1}, bp::ParticleCollection{Vector{Float64}}, 
                   o::Array{Float64, 1}, rng::MersenneTwister)::ParticleCollection{Vector{Float64}}
    weights = reweight(p.solver.PF, b, a, particles(bp), o)
    bw = WeightedParticleBelief(particles(bp), weights, sum(weights), nothing)
    posterior = resample(LowVarianceResampler(length(particles(b))), bw, p.solver.PF.predict_model.f,
                                              p.solver.PF.reweight_model.g, 
                                              b, a, o, rng) 
    return posterior::ParticleCollection{Vector{Float64}}
end

function posterior(p::Planner, 
                   b::ParticleCollection{Vector{Float64}}, 
                   a::Array{Float64, 1})::Tuple{ParticleCollection{Vector{Float64}},ParticleCollection{Vector{Float64}}, Vector{Float64}}
    bp = predict(p.solver.PF, b, a, p.pomdp.rng)
    x = rand(p.pomdp.rng, bp)
    o = SampleObservation(p, x)
    weights = reweight(p.solver.PF, b, a, bp, o)
    bw = WeightedParticleBelief(bp, weights, sum(weights), nothing)
    posterior = resample(LowVarianceResampler(length(particles(b))), bw, p.solver.PF.predict_model.f,
                                              p.solver.PF.reweight_model.g, 
                                              b, a, o, p.pomdp.rng)
    return (posterior, bp, o)::Tuple{ParticleCollection{Vector{Float64}},ParticleCollection{Vector{Float64}}, Vector{Float64}}
end


function _prob_safe(p::Planner, b::ParticleCollection{Vector{Float64}})::Float64

    local xᵒ::Vector{Float64}
    number_of_collided_states = 0.0 

    for x in particles(b)
        for i in 1:p.pomdp.obstacles_length
            xᵒ = p.pomdp.obstacles[i,:]
            if norm(x - xᵒ,2) < p.pomdp.obsRadii
                number_of_collided_states+=1.0
                break
            end
        end
    end
    prob_safe = 1. - number_of_collided_states/length(particles(b))
    return  prob_safe    
end


function _cvar_collision(p::Planner, b::ParticleCollection{Vector{Float64}})
    status = true
    for i in 1:p.pomdp.obstacles_length
        status = status && __cvar_collision_single_obstacle(p, b, p.pomdp.obstacles[i,:])
        if !status 
            return false
        end        
    end
    return true
end





function __cvar_collision_single_obstacle(p::Planner, 
                                          b::ParticleCollection{Vector{Float64}}, 
                                          xᵒ::Vector{Float64})::Bool

    distances = Float64[] 

    for x in particles(b)
        norm_with_obstacle_center = norm(x - xᵒ,2) 
        if norm_with_obstacle_center  > p.pomdp.obsRadii
            push!(distances, 0.0)
        else
            push!(distances, p.pomdp.obsRadii-norm_with_obstacle_center)
        end
    end
    @assert length(distances) == length(particles(b))
    α = 0.7 
    VaR = quantile(distances, 1-α)
    CVaR = 0.0
    i = 0
    for distance in distances
        if distance ≥ VaR
            CVaR += distance
            i+=1
        end    
    end     
    CVaR = CVaR/i
    if CVaR ≤ p.solver.δ
        return true
    else    
        return false
    end    
end

function _check_constraints_expected(p::Planner, probs::Vector{Float64})::Bool
    local ret_val::Bool
    mean(probs) ≥ 1 - p.solver.δ ?  ret_val=true : ret_val=false
    return ret_val
end


function _check_constraints_risk(p::Planner, probs::Vector{Float64})::Bool
    indicators = Bool[]
    for prob in probs
        if prob ≥ 1- p.solver.δ
            push!(indicators, true)
        else
            push!(indicators, false)
        end         
    end    
    local ret_val::Bool
    sum(indicators)/length(indicators) ≥ 1 - p.solver.ϵ ?  ret_val=true : ret_val=false
    return ret_val
end


function _check_constraints_cvar(p::Planner, this_level_constraints::Vector{Bool})::Bool
   
    m = length(this_level_constraints)
    
    prob = sum(this_level_constraints)/m    
    local ret_val::Bool
    prob ≥ 1 - p.solver.ϵ ?  ret_val=true : ret_val=false
    return ret_val
end


function _make_belief_safe!(pomdp::POMDPscenario , b::ParticleCollection{Vector{Float64}},rng::MersenneTwister)
    n = length(particles(b))
    while true
        try
            status = false
            safe_particles = Vector{Float64}[]
            norm_state_obstacles = zeros(pomdp.obstacles_length) 
            for xᵍ in particles(b)
                for i in 1:pomdp.obstacles_length
                    xᵒ = pomdp.obstacles[i,:]
                    norm_state_obstacles[i] = norm(xᵍ - xᵒ,2)   
                end    
                if minimum(norm_state_obstacles)  > pomdp.obsRadii
                    push!(safe_particles, xᵍ)
                end
            end  
            if length(safe_particles) < n
                status = true
                new_particles = sample(rng, safe_particles, n; replace=true, ordered=false)
                b.particles = new_particles
            end    
            return status
        catch
            printstyled("\ncan not fix belief\n", color=:red)
            b.particles = sample(rng, particles(b), n; replace=true, ordered=false)
        end
    end        
end    


function _mean_distance_to_goal(p::Planner, b::ParticleCollection{Vector{Float64}})::Float64
    r_b_x = 0.
    for x in particles(b)
        r_b_x -= norm(x - p.pomdp.goal,2) / length(particles(b))
    end
    return r_b_x::Float64
end


function _mean_distance_to_goal(p::Planner, b::FullNormal)::Float64
    return -(norm(b.μ-p.pomdp.goal,2) + tr(b.Σ))
end


function __calculate_obs_weight(pomdp::POMDPscenario , 
                                b::ParticleCollection{Vector{Float64}}, 
                                bsafe::ParticleCollection{Vector{Float64}}, o::Vector{Float64})
    zlikelihood = [likelihood(pomdp, x, o)/length(particles(b))  for x in particles(b)]
    z̄likelihood = [likelihood(pomdp, x, o)/length(particles(bsafe))  for x in particles(bsafe)]
    return sum(z̄likelihood)/sum(zlikelihood)                     
end    


