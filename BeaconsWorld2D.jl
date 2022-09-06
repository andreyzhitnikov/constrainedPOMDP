function BeaconsWorld2D(horizon::Int64, rng::MersenneTwister)
    d = 1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 0.0; 
               #2.0 0.0; 
               4.0 0.0;
               #6.0 0.0;
               8.0 0.0;
               #10.0 0.0;
               3.0 8.0;
               #10.0 4.0;
               10.0 6.0;
               #10.0 8.0;
               10.0 10.0;]
    obstacles = [5.0  5.0;
                 7.0  7.0;
                 3.0  3.0;]
    goal = [10, 10]
    a_space = [1.0  0.0;
              -1.0  0.0;
               0.0  1.0;
               0.0 -1.0;
               1/sqrt(2)  1/sqrt(2);
              -1/sqrt(2)  1/sqrt(2);
               1/sqrt(2) -1/sqrt(2);
              -1/sqrt(2) -1/sqrt(2);
               0.0  0.0             ]

    a_space_length = length(a_space[:,1])       
    obstacles_length = length(obstacles[:,1])
    σʷ = 0.1
    σᵛ = 0.01
    pomdp = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                                 σʷ = σʷ,
                                 σᵛ = σᵛ,   
                                 Σw=σʷ*[1.0 0.0; 0.0 1.0],
                                 Σv=σᵛ*[1.0 0.0; 0.0 1.0], 
                                 γ=0.99,
                                 Dmax = horizon,
                                 λ = 1,
                                 obsRadii = 0.5,
                                 goalRadii = 1.,
                                 goal = goal,
                                 rewardGoal = 10,
                                 rewardObs = -1,
                                 rng = rng, a_space=a_space, beacons=beacons, 
                                 a_space_length = a_space_length,
                                 obstacles_length = obstacles_length,
                                 obstacles=obstacles, d=d, rmin=rmin) 
    return pomdp
end

function update_observation_cov!(pomdp::POMDPscenario, x::Array{Float64, 1})
    mindist = Inf
    for i in 1:length(pomdp.beacons[:,1])
        distance = norm(x - pomdp.beacons[i,:])
        if distance <= pomdp.d
            pomdp.Σv = Matrix(Diagonal([1., 1.]))*pomdp.σᵛ
            return pomdp.Σv
        elseif distance < mindist
            mindist = distance
        end
    end
    # if no beacon is near by, get noise meas.
    pomdp.Σv = Matrix(Diagonal([1., 1.]))*pomdp.σʷ*mindist
    return pomdp.Σv
end

function dynamics(pomdp::POMDPscenario, x::Array{Float64, 1}, a::Array{Float64, 1}, rng::MersenneTwister)
    #global pomdp
    return SampleMotionModel(pomdp::POMDPscenario, a, x, rng)
end

function pdfObservationModel(pomdp::POMDPscenario, x_prev::Vector{Float64}, a::Vector{Float64}, x::Array{Float64, 1}, obs::Array{Float64, 1})
    #global pomdp::POMDPscenario
    (pomdp::POMDPscenario).Σv = update_observation_cov!((pomdp::POMDPscenario), x)
    Nv = MvNormal([0, 0], (pomdp::POMDPscenario).Σv)
    noise = obs - x
    return pdf(Nv, noise)
end

# input: belief at k, b(x_k), action a_k
# output: predicted gaussian belief b(xp)~N((μp,Σp)
function PropagateBelief(b::FullNormal, pomdp::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = pomdp.F
    Σw, Σv = pomdp.Σw, pomdp.Σv
    # calculations
    A = [ Σb^(-0.5) zeros(2,2); -Σw^(-0.5) Σw^(-0.5)]
    b = [ Σb^(-0.5)*μb; Σw^(-0.5)*a]
    # predict
    μp = inv(transpose(A)*A)*(transpose(A)*b)
    Σp = inv(transpose(A)*A) 
    μp = μp[3:4] # add your code here 
    Σp = Σp[3:4, 3:4] # add your code here 
    return MvNormal(μp, Σp)
end 

# input: belief at k, b(x_k), action a_k and observation z_k+1
# output: updated posterior gaussian belief b(x')~N(μb′, Σb′)
function PropagateUpdateBelief(b::FullNormal, pomdp::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = pomdp.F
    Σw, Σv = pomdp.Σw, pomdp.Σv
    # calculations
    A = [ Σb^(-0.5) zeros(2,2); -Σw^(-0.5) Σw^(-0.5); zeros(2,2) Σv^(-0.5)]
    b = [ Σb^(-0.5)*μb; Σw^(-0.5)*a; Σv^(-0.5)*o ]
    # predict
    μp = inv(transpose(A)*A)*(transpose(A)*b)
    Σp = inv(transpose(A)*A)
    # update
    # marginalize
    μb′ = μp[3:4]
    Σb′ = Σp[3:4, 3:4]
    return MvNormal(μb′, Σb′)
end

# input: state x and action a
# output: next state x'
function SampleMotionModel(pomdp::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1}, rng::MersenneTwister)
    Nw = MvNormal([0, 0], pomdp.Σw) # multivariate gaussian
    w = rand(rng, Nw)
    return x + a + w
end 

function pdfMotionModel(pomdp::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1}, x_prev::Array{Float64, 1})
    Nw = MvNormal([0, 0], pomdp.Σw)
    w = x - x_prev - a
    return pdf(Nw,w)
end 

# input: state x
# output: available observation z_rel and index, null otherwise
function GenerateObservationFromBeacons(pomdp::POMDPscenario, x::Array{Float64, 1}, fixed_cov::Bool, rng::MersenneTwister)::Union{NamedTuple, Nothing}
    distances = zeros(length(pomdp.beacons[:,1]))
    for index in 1:length(pomdp.beacons[:,1])
        distances[index] = norm(x - pomdp.beacons[index, :]) # calculate distances from x to all beacons
    end
    index = argmin(distances) # get observation only from nearest beacon
    pomdp.Σv = update_observation_cov!(pomdp, x)
    Nv = MvNormal([0, 0], pomdp.Σv)
    v = rand(rng, Nv)
    dX = x - pomdp.beacons[index, :]
    obs = dX + v 
    #pdf_val = pdf(Nv, v) #debug
    # println("value of obs1. $pdf_val, noise = $v, obs=$obs, dX=$dX, x=$x, beaconidx=$index, cov=$cov")
    return (obs=obs, index=index) 
end

function SampleObservation(p::Planner, x_propagated::Array{Float64, 1}, rng)
    o = GenerateObservationFromBeacons(p.pomdp, x_propagated, false, rng)
    return o[1] + p.pomdp.beacons[o[2], :]
end

function likelihood(pomdp::POMDPscenario, x::Vector{Float64},o::Vector{Float64})
    return pdfObservationModel(pomdp,[0.], [0.], x, o)
end

function initBelief()
    μ₀ = [0.0, 0.0]
    Σ₀ = [1.0 0.0; 0.0 1.0]
    return MvNormal(μ₀, Σ₀)
end

initState() = [-0.5, -0.2]


function oneStepSim(p::Planner, b::FullNormal, x_prev::Array{Float64, 1}, a::Array{Float64, 1})
    # create GT Trajectory, update horizon
    p.pomdp.Dmax -= 1

    b_prop = PropagateBelief(b, p.pomdp, a)
    x = SampleMotionModel(p.pomdp, a, x_prev, p.pomdp.rng)
    o_rel = GenerateObservationFromBeacons(p.pomdp, x, false, p.pomdp.rng)
    if o_rel === nothing
        o = nothing
        b_post = b_prop
    else
        o = o_rel[1] + p.pomdp.beacons[o_rel[2], :]
        # update Cov. according to distance from beacon
        update_observation_cov!(p.pomdp, x)
        b_post = PropagateUpdateBelief(b_prop, p.pomdp, a, o)
        
    end
    r = _mean_distance_to_goal(p, b_post)#reward(p, b_post, x)
    # check for collision 
    coll_status = false
    for i in 1:p.pomdp.obstacles_length
        x_o = p.pomdp.obstacles[i,:] # can check whole line connecting two points here 
        if norm(x - x_o,2) < p.pomdp.obsRadii
            coll_status = true
        end
    end
    return b_post, r, x, o, coll_status
end
